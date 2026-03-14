/**
 * @file numerical_utils.hpp
 * @brief Numerical utilities for precision and convergence
 */

#ifndef STATCPP_NUMERICAL_UTILS_HPP
#define STATCPP_NUMERICAL_UTILS_HPP

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace statcpp {

/**
 * @brief Machine epsilon for double precision
 */
constexpr double epsilon = std::numeric_limits<double>::epsilon();

/**
 * @brief Default relative tolerance for floating-point comparisons
 */
constexpr double default_rel_tol = 1e-9;

/**
 * @brief Default absolute tolerance for floating-point comparisons
 */
constexpr double default_abs_tol = 1e-12;

/**
 * @brief Check if two floating-point numbers are approximately equal
 *
 * Uses both relative and absolute tolerance:
 * |a - b| <= max(rel_tol * max(|a|, |b|), abs_tol)
 *
 * @param a First value
 * @param b Second value
 * @param rel_tol Relative tolerance (default: 1e-9)
 * @param abs_tol Absolute tolerance (default: 1e-12)
 * @return true if values are approximately equal
 */
inline bool approx_equal(double a, double b,
                         double rel_tol = default_rel_tol,
                         double abs_tol = default_abs_tol)
{
    // Handle exact equality (including infinities)
    if (a == b) {
        return true;
    }

    // Handle NaN
    if (std::isnan(a) || std::isnan(b)) {
        return false;
    }

    // Compute absolute difference
    double diff = std::abs(a - b);

    // Check absolute tolerance
    if (diff <= abs_tol) {
        return true;
    }

    // Check relative tolerance
    double max_abs = std::max(std::abs(a), std::abs(b));
    return diff <= rel_tol * max_abs;
}

/**
 * @brief Check if a value is close to zero
 *
 * @param x Value to check
 * @param tol Tolerance (default: 1e-12)
 * @return true if |x| <= tol
 */
inline bool is_zero(double x, double tol = default_abs_tol)
{
    return std::abs(x) <= tol;
}

/**
 * @brief Check if a value is finite (not infinity or NaN)
 *
 * @param x Value to check
 * @return true if x is finite
 */
inline bool is_finite(double x)
{
    return std::isfinite(x);
}

/**
 * @brief Check if all values in a range are finite
 *
 * @tparam Iterator RandomAccessIterator type
 * @param first Beginning of sequence
 * @param last End of sequence
 * @return true if all values are finite
 */
template <typename Iterator>
bool all_finite(Iterator first, Iterator last)
{
    for (auto it = first; it != last; ++it) {
        if (!std::isfinite(static_cast<double>(*it))) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Check for convergence based on absolute change
 *
 * Convergence criterion: |x_new - x_old| <= tol
 *
 * @param x_new New value
 * @param x_old Old value
 * @param tol Tolerance (default: 1e-6)
 * @return true if converged
 */
inline bool has_converged_abs(double x_new, double x_old,
                               double tol = 1e-6)
{
    return std::abs(x_new - x_old) <= tol;
}

/**
 * @brief Check for convergence based on relative change
 *
 * Convergence criterion: |x_new - x_old| / max(|x_old|, eps) <= tol
 *
 * @param x_new New value
 * @param x_old Old value
 * @param tol Tolerance (default: 1e-6)
 * @return true if converged
 */
inline bool has_converged_rel(double x_new, double x_old,
                               double tol = 1e-6)
{
    double denominator = std::max(std::abs(x_old), epsilon);
    return std::abs(x_new - x_old) / denominator <= tol;
}

/**
 * @brief Check for convergence using combined absolute and relative criteria
 *
 * Convergence criterion: |x_new - x_old| <= abs_tol + rel_tol * |x_old|
 *
 * @param x_new New value
 * @param x_old Old value
 * @param abs_tol Absolute tolerance (default: 1e-8)
 * @param rel_tol Relative tolerance (default: 1e-6)
 * @return true if converged
 */
inline bool has_converged(double x_new, double x_old,
                          double abs_tol = 1e-8,
                          double rel_tol = 1e-6)
{
    double threshold = abs_tol + rel_tol * std::abs(x_old);
    return std::abs(x_new - x_old) <= threshold;
}

/**
 * @brief Safely compute log(1 + x) for small x
 *
 * Uses std::log1p which is more accurate than log(1 + x) for small x
 *
 * @param x Input value
 * @return log(1 + x)
 */
inline double log1p_safe(double x)
{
    return std::log1p(x);
}

/**
 * @brief Safely compute exp(x) - 1 for small x
 *
 * Uses std::expm1 which is more accurate than exp(x) - 1 for small x
 *
 * @param x Input value
 * @return exp(x) - 1
 */
inline double expm1_safe(double x)
{
    return std::expm1(x);
}

/**
 * @brief Clamp a value to a range [min_val, max_val]
 *
 * @param x Value to clamp
 * @param min_val Minimum value
 * @param max_val Maximum value
 * @return Clamped value
 * @throw std::invalid_argument if min_val > max_val
 */
inline double clamp(double x, double min_val, double max_val)
{
    if (min_val > max_val) {
        throw std::invalid_argument("statcpp::clamp: min_val must be <= max_val");
    }
    return std::max(min_val, std::min(x, max_val));
}

/**
 * @brief Check if a value is in range [min_val, max_val]
 *
 * @param x Value to check
 * @param min_val Minimum value
 * @param max_val Maximum value
 * @return true if min_val <= x <= max_val
 */
inline bool in_range(double x, double min_val, double max_val)
{
    return x >= min_val && x <= max_val;
}

/**
 * @brief Compute the relative error between two values
 *
 * relative_error = |x - x_ref| / max(|x_ref|, eps)
 *
 * @param x Computed value
 * @param x_ref Reference value
 * @return Relative error
 */
inline double relative_error(double x, double x_ref)
{
    double denominator = std::max(std::abs(x_ref), epsilon);
    return std::abs(x - x_ref) / denominator;
}

/**
 * @brief Safe division with check for division by zero
 *
 * @param numerator Numerator
 * @param denominator Denominator
 * @param default_value Value to return if denominator is zero (default: NaN)
 * @return numerator / denominator, or default_value if denominator is zero
 */
inline double safe_divide(double numerator, double denominator,
                          double default_value = std::numeric_limits<double>::quiet_NaN())
{
    if (is_zero(denominator)) {
        return default_value;
    }
    return numerator / denominator;
}

/**
 * @brief Compute sum with Kahan summation algorithm
 *
 * Kahan summation reduces numerical error in summing a sequence of floating-point numbers.
 * More accurate than naive summation for large sequences or when values vary widely in magnitude.
 *
 * @tparam Iterator RandomAccessIterator type
 * @param first Beginning of sequence
 * @param last End of sequence
 * @return Sum of elements using Kahan algorithm
 */
template <typename Iterator>
double kahan_sum(Iterator first, Iterator last)
{
    double sum = 0.0;
    double compensation = 0.0;  // Running compensation for lost low-order bits

    for (auto it = first; it != last; ++it) {
        double value = static_cast<double>(*it);
        double y = value - compensation;
        double t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }

    return sum;
}

/**
 * @brief Compute sum with Kahan summation algorithm (with projection)
 */
template <typename Iterator, typename Proj>
double kahan_sum(Iterator first, Iterator last, Proj proj)
{
    double sum = 0.0;
    double compensation = 0.0;

    for (auto it = first; it != last; ++it) {
        double value = static_cast<double>(proj(*it));
        double y = value - compensation;
        double t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }

    return sum;
}

/**
 * @brief Check if two ranges are approximately equal element-wise
 *
 * @tparam Iterator1 RandomAccessIterator type for first sequence
 * @tparam Iterator2 RandomAccessIterator type for second sequence
 * @param first1 Beginning of first sequence
 * @param last1 End of first sequence
 * @param first2 Beginning of second sequence
 * @param last2 End of second sequence
 * @param rel_tol Relative tolerance (default: 1e-9)
 * @param abs_tol Absolute tolerance (default: 1e-12)
 * @return true if all corresponding elements are approximately equal
 */
template <typename Iterator1, typename Iterator2>
bool approx_equal_range(Iterator1 first1, Iterator1 last1,
                        Iterator2 first2, Iterator2 last2,
                        double rel_tol = default_rel_tol,
                        double abs_tol = default_abs_tol)
{
    auto n1 = std::distance(first1, last1);
    auto n2 = std::distance(first2, last2);

    if (n1 != n2) {
        return false;
    }

    auto it1 = first1;
    auto it2 = first2;

    while (it1 != last1) {
        double v1 = static_cast<double>(*it1);
        double v2 = static_cast<double>(*it2);

        if (!approx_equal(v1, v2, rel_tol, abs_tol)) {
            return false;
        }

        ++it1;
        ++it2;
    }

    return true;
}

} // namespace statcpp

#endif // STATCPP_NUMERICAL_UTILS_HPP
