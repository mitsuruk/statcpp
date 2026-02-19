/**
 * @file survival.hpp
 * @brief Survival analysis functions
 *
 * Provides functions for survival analysis including Kaplan-Meier estimation,
 * Log-rank test, and Nelson-Aalen estimation.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "statcpp/continuous_distributions.hpp"

namespace statcpp {

// ============================================================================
// Kaplan-Meier Estimator
// ============================================================================

/**
 * @brief Kaplan-Meier estimation result
 *
 * Holds the survival curve estimation results.
 */
struct kaplan_meier_result {
    std::vector<double> times;              ///< Event times
    std::vector<double> survival;           ///< Survival probabilities
    std::vector<double> se;                 ///< Standard errors (Greenwood's formula)
    std::vector<double> ci_lower;           ///< 95% confidence interval lower bound
    std::vector<double> ci_upper;           ///< 95% confidence interval upper bound
    std::vector<std::size_t> n_at_risk;     ///< Risk set size
    std::vector<std::size_t> n_events;      ///< Number of events at each time point
    std::vector<std::size_t> n_censored;    ///< Number of censored at each time point
};

/**
 * @brief Estimate Kaplan-Meier survival curve
 *
 * Estimates the survival curve from survival time data including censored observations.
 * Calculates standard errors using Greenwood's formula and provides 95% confidence intervals.
 *
 * @param times Vector of observation times
 * @param events Event occurrence flags (true = event occurred, false = censored)
 * @return Kaplan-Meier estimation result
 * @throws std::invalid_argument If times and events have different sizes or data is empty
 */
inline kaplan_meier_result kaplan_meier(
    const std::vector<double>& times,
    const std::vector<bool>& events)
{
    if (times.size() != events.size()) {
        throw std::invalid_argument("statcpp::kaplan_meier: times and events must have same length");
    }
    if (times.empty()) {
        throw std::invalid_argument("statcpp::kaplan_meier: empty data");
    }

    std::size_t n = times.size();

    // Sort indices by time
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&times](std::size_t i, std::size_t j) {
                  return times[i] < times[j];
              });

    kaplan_meier_result result;
    result.times.push_back(0.0);
    result.survival.push_back(1.0);
    result.se.push_back(0.0);
    result.n_at_risk.push_back(n);
    result.n_events.push_back(0);
    result.n_censored.push_back(0);

    double S = 1.0;  // Cumulative survival probability
    double var_sum = 0.0;  // Cumulative sum of Greenwood variance terms
    std::size_t n_risk = n;  // Risk set size

    std::size_t i = 0;
    while (i < n) {
        double t = times[indices[i]];

        // Count events and censored at this time point
        std::size_t d = 0;  // Number of events
        std::size_t c = 0;  // Number of censored

        while (i < n && times[indices[i]] == t) {
            if (events[indices[i]]) {
                d++;
            } else {
                c++;
            }
            i++;
        }

        // Update survival probability only if there were events
        if (d > 0) {
            double q = static_cast<double>(d) / static_cast<double>(n_risk);
            S *= (1.0 - q);

            // Greenwood variance
            if (n_risk > d) {
                var_sum += static_cast<double>(d) /
                          (static_cast<double>(n_risk) * static_cast<double>(n_risk - d));
            }

            double se = S * std::sqrt(var_sum);

            // 95% confidence interval (log transformation)
            double z = 1.96;
            double ci_lower, ci_upper;
            if (S > 0 && S < 1) {
                double log_S = std::log(S);
                double se_log = se / S;
                ci_lower = std::exp(log_S - z * se_log);
                ci_upper = std::exp(log_S + z * se_log);
                ci_lower = std::max(0.0, ci_lower);
                ci_upper = std::min(1.0, ci_upper);
            } else {
                ci_lower = S;
                ci_upper = S;
            }

            result.times.push_back(t);
            result.survival.push_back(S);
            result.se.push_back(se);
            result.ci_lower.push_back(ci_lower);
            result.ci_upper.push_back(ci_upper);
            result.n_at_risk.push_back(n_risk);
            result.n_events.push_back(d);
            result.n_censored.push_back(c);
        }

        n_risk -= (d + c);
    }

    // Set confidence interval for the first point
    result.ci_lower.insert(result.ci_lower.begin(), 1.0);
    result.ci_upper.insert(result.ci_upper.begin(), 1.0);

    return result;
}

// ============================================================================
// Log-rank Test
// ============================================================================

/**
 * @brief Log-rank test result
 *
 * Holds the results of comparing survival curves between two groups.
 */
struct logrank_result {
    double statistic;     ///< Test statistic (chi-square)
    double p_value;       ///< p-value
    std::size_t df;       ///< Degrees of freedom
    double expected1;     ///< Expected number of events in group 1
    double expected2;     ///< Expected number of events in group 2
    std::size_t observed1; ///< Observed number of events in group 1
    std::size_t observed2; ///< Observed number of events in group 2
};

/**
 * @brief Log-rank test (comparison of two survival curves)
 *
 * Tests whether the survival curves of two groups are equal.
 * A nonparametric test that considers the risk set at each time point.
 *
 * @param times1 Observation times for group 1
 * @param events1 Event occurrence flags for group 1
 * @param times2 Observation times for group 2
 * @param events2 Event occurrence flags for group 2
 * @return Log-rank test result
 * @throws std::invalid_argument If times and events sizes don't match or data is empty
 */
inline logrank_result logrank_test(
    const std::vector<double>& times1,
    const std::vector<bool>& events1,
    const std::vector<double>& times2,
    const std::vector<bool>& events2)
{
    if (times1.size() != events1.size() || times2.size() != events2.size()) {
        throw std::invalid_argument("statcpp::logrank_test: times and events must have same length");
    }
    if (times1.empty() || times2.empty()) {
        throw std::invalid_argument("statcpp::logrank_test: empty data");
    }

    // Get all unique times and sort
    std::vector<double> all_times;
    all_times.reserve(times1.size() + times2.size());
    for (double t : times1) all_times.push_back(t);
    for (double t : times2) all_times.push_back(t);
    std::sort(all_times.begin(), all_times.end());
    all_times.erase(std::unique(all_times.begin(), all_times.end()), all_times.end());

    // Calculate risk set and event count for each time point
    std::size_t O1 = 0;  // Observed events in group 1
    std::size_t O2 = 0;  // Observed events in group 2
    double E1 = 0.0;     // Expected events in group 1
    double var = 0.0;    // Variance

    // Check if events occurred at each time
    for (double t : all_times) {
        // Number of observations >= this time = risk set
        std::size_t n1_risk = 0;
        std::size_t n2_risk = 0;
        std::size_t d1 = 0;  // Events in group 1
        std::size_t d2 = 0;  // Events in group 2

        for (std::size_t i = 0; i < times1.size(); ++i) {
            if (times1[i] >= t) {
                n1_risk++;
            }
            if (times1[i] == t && events1[i]) {
                d1++;
            }
        }

        for (std::size_t i = 0; i < times2.size(); ++i) {
            if (times2[i] >= t) {
                n2_risk++;
            }
            if (times2[i] == t && events2[i]) {
                d2++;
            }
        }

        std::size_t d = d1 + d2;  // Total events
        std::size_t n_risk = n1_risk + n2_risk;  // Total at risk

        if (d > 0 && n_risk > 0) {
            O1 += d1;
            O2 += d2;

            double e1 = static_cast<double>(n1_risk) * static_cast<double>(d) /
                       static_cast<double>(n_risk);
            E1 += e1;

            // Variance (hypergeometric distribution variance)
            if (n_risk > 1) {
                var += static_cast<double>(n1_risk) * static_cast<double>(n2_risk) *
                       static_cast<double>(d) * static_cast<double>(n_risk - d) /
                       (static_cast<double>(n_risk) * static_cast<double>(n_risk) *
                        static_cast<double>(n_risk - 1));
            }
        }
    }

    double E2 = static_cast<double>(O1 + O2) - E1;

    // Test statistic
    double stat = 0.0;
    if (var > 0) {
        double diff = static_cast<double>(O1) - E1;
        stat = (diff * diff) / var;
    }

    // p-value (chi-square distribution, df=1)
    double p_value = 1.0 - statcpp::chisq_cdf(stat, 1.0);

    return {stat, p_value, 1, E1, E2, O1, O2};
}

// ============================================================================
// Median Survival Time
// ============================================================================

/**
 * @brief Calculate median survival time
 *
 * Returns the time at which survival probability reaches 50%.
 * Returns NaN if 50% is not reached.
 *
 * @param km Kaplan-Meier estimation result
 * @return Median survival time (NaN if 50% is not reached)
 */
inline double median_survival_time(const kaplan_meier_result& km)
{
    // Find the first time where S(t) becomes <= 0.5
    for (std::size_t i = 0; i < km.survival.size(); ++i) {
        if (km.survival[i] <= 0.5) {
            return km.times[i];
        }
    }
    // Return NaN if 50% is not reached
    return std::numeric_limits<double>::quiet_NaN();
}

// ============================================================================
// Hazard Rate (Actuarial Method)
// ============================================================================

/**
 * @brief Hazard rate result
 *
 * Holds the hazard rate and cumulative hazard at each time point.
 */
struct hazard_rate_result {
    std::vector<double> times;          ///< Interval start times
    std::vector<double> hazard;         ///< Hazard rates
    std::vector<double> cumulative_hazard;  ///< Cumulative hazard
};

/**
 * @brief Nelson-Aalen cumulative hazard estimation
 *
 * Estimates the cumulative hazard function H(t) using the Nelson-Aalen estimator.
 * Handles survival time data with censoring.
 *
 * @note The Nelson-Aalen estimator is a nonparametric estimator of the cumulative hazard function:
 *       H(t) = sum_{t_i <= t} d_i / n_i
 *       where d_i is the number of events at time t_i and n_i is the risk set size.
 *
 *       Relationship between cumulative hazard and survival function:
 *       - S(t) = exp(-H(t)) (approximation via exponential transformation)
 *       - Kaplan-Meier estimator directly estimates S(t) = prod(1 - d_i/n_i)
 *       - For small event probabilities, both are approximately equal
 *
 *       Advantages of Nelson-Aalen estimation:
 *       - Simpler confidence interval construction (variance estimated by sum d_i/n_i^2)
 *       - Consistent with Cox regression baseline hazard estimation (Breslow estimator)
 *
 * @param times Vector of observation times
 * @param events Event occurrence flags (true = event occurred, false = censored)
 * @return Hazard rate estimation result
 * @throws std::invalid_argument If times and events have different sizes or data is empty
 */
inline hazard_rate_result nelson_aalen(
    const std::vector<double>& times,
    const std::vector<bool>& events)
{
    if (times.size() != events.size()) {
        throw std::invalid_argument("statcpp::nelson_aalen: times and events must have same length");
    }
    if (times.empty()) {
        throw std::invalid_argument("statcpp::nelson_aalen: empty data");
    }

    std::size_t n = times.size();

    // Sort indices by time
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&times](std::size_t i, std::size_t j) {
                  return times[i] < times[j];
              });

    hazard_rate_result result;
    result.times.push_back(0.0);
    result.hazard.push_back(0.0);
    result.cumulative_hazard.push_back(0.0);

    double H = 0.0;  // Cumulative hazard
    std::size_t n_risk = n;

    std::size_t i = 0;
    while (i < n) {
        double t = times[indices[i]];
        std::size_t d = 0;
        std::size_t c = 0;

        while (i < n && times[indices[i]] == t) {
            if (events[indices[i]]) {
                d++;
            } else {
                c++;
            }
            i++;
        }

        if (d > 0 && n_risk > 0) {
            double h = static_cast<double>(d) / static_cast<double>(n_risk);
            H += h;

            result.times.push_back(t);
            result.hazard.push_back(h);
            result.cumulative_hazard.push_back(H);
        }

        n_risk -= (d + c);
    }

    return result;
}

} // namespace statcpp
