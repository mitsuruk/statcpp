/**
 * @file nan_utils.hpp
 * @brief NaN (missing value) handling helpers
 *
 * Internal helpers shared by the public functions to implement the NaN policy
 * (docs/NAN_POLICY.md): detecting NaN, removing it, ordering it last, and
 * rejecting it with a uniform error message.
 *
 * Everything in this file lives in statcpp::detail and is not part of the
 * public API.
 */

#pragma once

#include <cmath>
#include <functional>
#include <cstddef>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace statcpp {
namespace detail {

// ============================================================================
// Detection
// ============================================================================

/**
 * @brief Check whether a value is NaN
 *
 * Floating-point values are tested with std::isnan. Any other type cannot hold
 * NaN, so the result is always false; this lets template functions over a
 * generic value type (for example value_counts<T>) use the same check.
 *
 * @tparam T Value type
 * @param value Value to test
 * @return true if value is a floating-point NaN
 */
template <typename T>
bool is_nan_value(const T& value)
{
    if constexpr (std::is_floating_point_v<T>) {
        return std::isnan(value);
    } else {
        (void)value;
        return false;
    }
}

/**
 * @brief Check whether a range contains NaN
 *
 * @tparam Iterator Iterator type
 * @param first Beginning of range
 * @param last End of range
 * @return true if any element is NaN
 */
template <typename Iterator>
bool has_nan(Iterator first, Iterator last)
{
    for (; first != last; ++first) {
        if (is_nan_value(*first)) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Check whether a range contains NaN after projection
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Beginning of range
 * @param last End of range
 * @param proj Projection applied to each element
 * @return true if any projected value is NaN
 */
template <typename Iterator, typename Projection>
bool has_nan(Iterator first, Iterator last, Projection proj)
{
    for (; first != last; ++first) {
        if (is_nan_value(std::invoke(proj, *first))) {
            return true;
        }
    }
    return false;
}

// ============================================================================
// Removal
// ============================================================================

/**
 * @brief Copy a range without its NaN elements
 *
 * The relative order of the remaining elements is preserved. This implements
 * the "remove NA, then compute" behaviour of R functions such as t.test.
 *
 * @tparam Iterator Iterator type
 * @param first Beginning of range
 * @param last End of range
 * @return The non-NaN elements, in their original order
 */
template <typename Iterator>
auto drop_nan(Iterator first, Iterator last)
{
    using value_type = std::remove_cv_t<typename std::iterator_traits<Iterator>::value_type>;
    std::vector<value_type> result;
    for (; first != last; ++first) {
        if (!is_nan_value(*first)) {
            result.push_back(*first);
        }
    }
    return result;
}

/**
 * @brief Project a range and copy the projected values that are not NaN
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Beginning of range
 * @param last End of range
 * @param proj Projection applied to each element
 * @return The non-NaN projected values, in their original order
 */
template <typename Iterator, typename Projection>
auto drop_nan(Iterator first, Iterator last, Projection proj)
{
    using projected_type = std::decay_t<
        std::invoke_result_t<Projection&, typename std::iterator_traits<Iterator>::reference>>;
    std::vector<projected_type> result;
    for (; first != last; ++first) {
        auto value = std::invoke(proj, *first);
        if (!is_nan_value(value)) {
            result.push_back(value);
        }
    }
    return result;
}

/**
 * @brief Copy two parallel ranges, dropping every pair that contains NaN
 *
 * A pair is dropped when either element is NaN, as R's complete.cases does for
 * paired data. The second range must have at least as many elements as the
 * first; only the first std::distance(x_first, x_last) elements are read.
 *
 * @tparam IteratorX Iterator type of the first range
 * @tparam IteratorY Iterator type of the second range
 * @param x_first Beginning of the first range
 * @param x_last End of the first range
 * @param y_first Beginning of the second range
 * @return The complete pairs, as two vectors of equal length
 */
template <typename IteratorX, typename IteratorY>
auto drop_nan_pairs(IteratorX x_first, IteratorX x_last, IteratorY y_first)
{
    using x_type = std::remove_cv_t<typename std::iterator_traits<IteratorX>::value_type>;
    using y_type = std::remove_cv_t<typename std::iterator_traits<IteratorY>::value_type>;
    std::pair<std::vector<x_type>, std::vector<y_type>> result;
    for (; x_first != x_last; ++x_first, ++y_first) {
        if (!is_nan_value(*x_first) && !is_nan_value(*y_first)) {
            result.first.push_back(*x_first);
            result.second.push_back(*y_first);
        }
    }
    return result;
}

/**
 * @brief Apply an element-wise transform to the non-NaN values only
 *
 * The NaN elements are removed, func is applied to the rest, and the results are
 * written back to their original positions; NaN positions stay NaN. This is how
 * R's p.adjust() treats NA: the adjustment counts only the non-NA p-values.
 *
 * @tparam Function Callable taking const std::vector<double>& and returning a
 *         std::vector<double> of the same length
 * @param values Input values, possibly containing NaN
 * @param func Transform applied to the non-NaN values
 * @return The transformed values at the non-NaN positions, NaN elsewhere
 */
template <typename Function>
std::vector<double> map_non_nan(const std::vector<double>& values, Function func)
{
    std::vector<double> non_nan;
    std::vector<std::size_t> positions;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isnan(values[i])) {
            non_nan.push_back(values[i]);
            positions.push_back(i);
        }
    }
    std::vector<double> result(values.size(), std::numeric_limits<double>::quiet_NaN());
    if (non_nan.empty()) {
        return result;
    }
    const std::vector<double> mapped = func(non_nan);
    for (std::size_t k = 0; k < positions.size(); ++k) {
        result[positions[k]] = mapped[k];
    }
    return result;
}

// ============================================================================
// Matrices (rows = observations)
// ============================================================================

/**
 * @brief Check whether a matrix contains NaN
 *
 * @param matrix Matrix as a vector of rows
 * @return true if any element is NaN
 */
inline bool has_nan_matrix(const std::vector<std::vector<double>>& matrix)
{
    for (const auto& row : matrix) {
        if (has_nan(row.begin(), row.end())) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Check whether a design matrix or its response contains NaN
 *
 * @param X Predictor matrix (rows = observations)
 * @param y Response vector
 * @return true if any element of X or y is NaN
 */
inline bool has_nan_rows(const std::vector<std::vector<double>>& X, const std::vector<double>& y)
{
    return has_nan_matrix(X) || has_nan(y.begin(), y.end());
}

/**
 * @brief Copy the observations (rows of X and elements of y) that contain no NaN
 *
 * An observation is dropped when its response or any of its predictors is NaN,
 * as R's model functions do with na.action = na.omit. X and y must have the
 * same number of rows.
 *
 * @param X Predictor matrix (rows = observations)
 * @param y Response vector
 * @return The complete observations, as a matrix and a response vector
 */
inline std::pair<std::vector<std::vector<double>>, std::vector<double>> drop_nan_rows(
    const std::vector<std::vector<double>>& X, const std::vector<double>& y)
{
    std::pair<std::vector<std::vector<double>>, std::vector<double>> result;
    for (std::size_t i = 0; i < X.size(); ++i) {
        if (!has_nan(X[i].begin(), X[i].end()) && !is_nan_value(y[i])) {
            result.first.push_back(X[i]);
            result.second.push_back(y[i]);
        }
    }
    return result;
}

/**
 * @brief Copy the rows of a matrix that contain no NaN
 *
 * @param X Matrix (rows = observations)
 * @return The complete rows, in their original order
 */
inline std::vector<std::vector<double>> drop_nan_rows(const std::vector<std::vector<double>>& X)
{
    std::vector<std::vector<double>> result;
    for (const auto& row : X) {
        if (!has_nan(row.begin(), row.end())) {
            result.push_back(row);
        }
    }
    return result;
}

/**
 * @brief Throw if a matrix contains NaN
 *
 * @param matrix Matrix as a vector of rows
 * @param function_name Public function name, used in the message
 * @throws std::invalid_argument "statcpp::<function_name>: data contains NaN"
 */
inline void require_no_nan_matrix(const std::vector<std::vector<double>>& matrix, const char* function_name)
{
    if (has_nan_matrix(matrix)) {
        throw std::invalid_argument(std::string("statcpp::") + function_name + ": data contains NaN");
    }
}

// ============================================================================
// Ordering
// ============================================================================

/**
 * @brief Ascending order that places NaN last
 *
 * A strict weak ordering even when NaN is present (operator< alone is not, which
 * makes std::sort undefined behaviour): non-NaN values compare as usual, every
 * NaN compares greater than any non-NaN value, and NaN values are equivalent to
 * each other. Use std::stable_sort to keep NaN in their original order, as R's
 * order(na.last = TRUE) does.
 */
struct nan_last_less {
    /**
     * @brief Compare two values
     * @param a Left-hand value
     * @param b Right-hand value
     * @return true if a is ordered before b
     */
    template <typename T>
    bool operator()(const T& a, const T& b) const
    {
        const bool a_nan = is_nan_value(a);
        const bool b_nan = is_nan_value(b);
        if (a_nan || b_nan) {
            return !a_nan && b_nan;
        }
        return a < b;
    }
};

/**
 * @brief Descending order that places NaN last
 *
 * The descending counterpart of nan_last_less: NaN still comes last, as R's
 * order(decreasing = TRUE, na.last = TRUE) does.
 */
struct nan_last_greater {
    /**
     * @brief Compare two values
     * @param a Left-hand value
     * @param b Right-hand value
     * @return true if a is ordered before b
     */
    template <typename T>
    bool operator()(const T& a, const T& b) const
    {
        const bool a_nan = is_nan_value(a);
        const bool b_nan = is_nan_value(b);
        if (a_nan || b_nan) {
            return !a_nan && b_nan;
        }
        return a > b;
    }
};

// ============================================================================
// Rejection
// ============================================================================

/**
 * @brief Throw if a data range contains NaN
 *
 * Used by functions that cannot represent a missing result (integer or
 * categorical outputs) and by functions whose R counterpart raises an error.
 *
 * @tparam Iterator Iterator type
 * @param first Beginning of range
 * @param last End of range
 * @param function_name Public function name, used in the message
 * @throws std::invalid_argument "statcpp::<function_name>: data contains NaN"
 */
template <typename Iterator>
void require_no_nan(Iterator first, Iterator last, const char* function_name)
{
    if (has_nan(first, last)) {
        throw std::invalid_argument(std::string("statcpp::") + function_name + ": data contains NaN");
    }
}

/**
 * @brief Throw if a parameter is NaN
 *
 * A NaN parameter slips through range checks such as `sigma <= 0`, because every
 * comparison with NaN is false; call this before those checks.
 *
 * @param value Parameter value
 * @param function_name Public function name, used in the message
 * @param parameter_name Parameter name, used in the message
 * @throws std::invalid_argument "statcpp::<function_name>: <parameter_name> is NaN"
 */
inline void require_param_not_nan(double value, const char* function_name, const char* parameter_name)
{
    if (std::isnan(value)) {
        throw std::invalid_argument(std::string("statcpp::") + function_name + ": " + parameter_name +
                                    " is NaN");
    }
}

} // namespace detail
} // namespace statcpp
