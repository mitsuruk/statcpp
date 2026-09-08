/**
 * @file test_vs_r_descriptive.cpp
 * @brief Compare statcpp descriptive statistics against R 4.4.2 reference values.
 *
 * Covers basic_statistics.hpp, dispersion_spread.hpp, order_statistics.hpp,
 * shape_of_distribution.hpp, correlation_covariance.hpp, distance_metrics.hpp
 * and frequency_distribution.hpp.
 *
 * This group carries the most definition differences against R. Each reference
 * struct records which R form it was compared with, and the notable traps are
 * repeated next to the assertions here.
 */

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <map>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "statcpp/basic_statistics.hpp"
#include "statcpp/correlation_covariance.hpp"
#include "statcpp/dispersion_spread.hpp"
#include "statcpp/distance_metrics.hpp"
#include "statcpp/frequency_distribution.hpp"
#include "statcpp/order_statistics.hpp"
#include "statcpp/shape_of_distribution.hpp"

#include "r_compare.hpp"
#include "r_reference_descriptive.hpp"

namespace {

namespace ref = statcpp_test::r_ref;
using statcpp_test::MakeMatrix;

/// @brief Copy a reference array into the vector type the statcpp API expects.
template <std::size_t N>
std::vector<double> AsVector(const double (&values)[N])
{
    return std::vector<double>(values, values + N);
}

// ============================================================================
// basic_statistics.hpp
// ============================================================================

/// @brief Sums, means and index lookups against their R counterparts.
TEST(RBasicStatistics, Aggregates) {
    using Case = ref::BasicStatistics;
    const auto x = AsVector(Case::kX);
    const auto xs = AsVector(Case::kXSorted);
    EXPECT_R_FIELD(statcpp::sum(x.begin(), x.end()), Case, kSum);
    EXPECT_R_FIELD(statcpp::mean(x.begin(), x.end()), Case, kMean);
    // median and trimmed_mean assume a sorted range.
    EXPECT_R_FIELD(statcpp::median(xs.begin(), xs.end()), Case, kMedian);
    EXPECT_R_FIELD(statcpp::trimmed_mean(xs.begin(), xs.end(), Case::kTrim), Case, kTrimmedMean);
    EXPECT_R_FIELD(statcpp::geometric_mean(x.begin(), x.end()), Case, kGeometricMean);
    EXPECT_R_FIELD(statcpp::harmonic_mean(x.begin(), x.end()), Case, kHarmonicMean);
    // R's which.max/which.min are one-based; statcpp returns a zero-based index.
    EXPECT_R_FIELD(static_cast<double>(statcpp::argmax(x.begin(), x.end())), Case, kArgmax);
    EXPECT_R_FIELD(static_cast<double>(statcpp::argmin(x.begin(), x.end())), Case, kArgmin);
}

/// @brief Mode of a sample with a single most frequent value.
TEST(RBasicStatistics, Mode) {
    using Case = ref::Mode;
    const auto x = AsVector(Case::kX);
    EXPECT_R_FIELD(statcpp::mode(x.begin(), x.end()), Case, kMode);
}

/// @brief All modes of a sample where two values tie.
TEST(RBasicStatistics, Modes) {
    using Case = ref::Modes;
    const auto x = AsVector(Case::kX);
    auto modes = statcpp::modes(x.begin(), x.end());
    std::sort(modes.begin(), modes.end());
    EXPECT_R_RANGE(modes, Case, kModes);
}

/// @brief Weighted means against weighted.mean and the harmonic closed form.
TEST(RBasicStatistics, WeightedMeans) {
    using Case = ref::WeightedBasicStatistics;
    const auto x = AsVector(Case::kX);
    const auto w = AsVector(Case::kW);
    EXPECT_R_FIELD(statcpp::weighted_mean(x.begin(), x.end(), w.begin(), w.end()), Case,
                   kWeightedMean);
    EXPECT_R_FIELD(statcpp::weighted_harmonic_mean(x.begin(), x.end(), w.begin(), w.end()), Case,
                   kWeightedHarmonicMean);
}

/// @brief Logarithmic mean of two values.
TEST(RBasicStatistics, LogarithmicMean) {
    using Case = ref::LogarithmicMean;
    EXPECT_R_FIELD(statcpp::logarithmic_mean(Case::kA, Case::kB), Case, kY);
}

// ============================================================================
// dispersion_spread.hpp
// ============================================================================

/// @brief The variance and standard deviation family, including the ddof forms.
///
/// statcpp's var/stdev default to ddof = 0, which is the population form. R's
/// var/sd are the sample form, so the two bare calls do not correspond.
TEST(RDispersion, VarianceFamily) {
    using Case = ref::Dispersion;
    const auto x = AsVector(Case::kX);
    const auto b = x.begin();
    const auto e = x.end();
    EXPECT_R_FIELD(statcpp::var(b, e), Case, kVarDdof0);
    EXPECT_R_FIELD(statcpp::var(b, e, std::size_t{1}), Case, kVarDdof1);
    EXPECT_R_FIELD(statcpp::variance(b, e), Case, kVariance);
    EXPECT_R_FIELD(statcpp::sample_variance(b, e), Case, kSampleVariance);
    EXPECT_R_FIELD(statcpp::population_variance(b, e), Case, kPopulationVariance);
    EXPECT_R_FIELD(statcpp::stdev(b, e), Case, kStdevDdof0);
    EXPECT_R_FIELD(statcpp::stdev(b, e, std::size_t{1}), Case, kStdevDdof1);
    EXPECT_R_FIELD(statcpp::stddev(b, e), Case, kStddev);
    EXPECT_R_FIELD(statcpp::sample_stddev(b, e), Case, kSampleStddev);
    EXPECT_R_FIELD(statcpp::population_stddev(b, e), Case, kPopulationStddev);
}

/// @brief Spread measures that are not part of the variance family.
///
/// mean_absolute_deviation is the mean absolute deviation about the mean. R's
/// mad() is the median absolute deviation and is a different statistic.
TEST(RDispersion, SpreadMeasures) {
    using Case = ref::Dispersion;
    const auto x = AsVector(Case::kX);
    const auto xs = AsVector(Case::kXSorted);
    EXPECT_R_FIELD(statcpp::range(x.begin(), x.end()), Case, kRange);
    EXPECT_R_FIELD(statcpp::iqr(xs.begin(), xs.end()), Case, kIqr);
    EXPECT_R_FIELD(statcpp::mean_absolute_deviation(x.begin(), x.end()), Case,
                   kMeanAbsoluteDeviation);
    EXPECT_R_FIELD(statcpp::coefficient_of_variation(x.begin(), x.end()), Case,
                   kCoefficientOfVariation);
    EXPECT_R_FIELD(statcpp::geometric_stddev(x.begin(), x.end()), Case, kGeometricStddev);
}

/// @brief Weighted variance and standard deviation against cov.wt.
TEST(RDispersion, Weighted) {
    using Case = ref::WeightedDispersion;
    const auto x = AsVector(Case::kX);
    const auto w = AsVector(Case::kW);
    EXPECT_R_FIELD(statcpp::weighted_variance(x.begin(), x.end(), w.begin(), w.end()), Case,
                   kWeightedVariance);
    EXPECT_R_FIELD(statcpp::weighted_stddev(x.begin(), x.end(), w.begin(), w.end()), Case,
                   kWeightedStddev);
}

// ============================================================================
// order_statistics.hpp
// ============================================================================

/// @brief Percentiles swept over the whole unit interval against quantile(type = 7).
TEST(ROrderStatistics, PercentileSweep) {
    using Case = ref::Percentile;
    const auto x = AsVector(Case::kX);
    std::vector<double> actual;
    actual.reserve(std::size(Case::kP));
    for (const double p : Case::kP) actual.push_back(statcpp::percentile(x.begin(), x.end(), p));
    EXPECT_R_RANGE(actual, Case, kY);
}

/// @brief Extremes, quartiles and the five-number summary.
///
/// five_number_summary uses type 7 quantiles, not R's fivenum(), which reports
/// Tukey hinges.
TEST(ROrderStatistics, Summary) {
    using Case = ref::OrderStatistics;
    const auto x = AsVector(Case::kX);
    EXPECT_R_FIELD(statcpp::minimum(x.begin(), x.end()), Case, kMinimum);
    EXPECT_R_FIELD(statcpp::maximum(x.begin(), x.end()), Case, kMaximum);
    const auto q = statcpp::quartiles(x.begin(), x.end());
    EXPECT_R_FIELD(q.q1, Case, kQ1);
    EXPECT_R_FIELD(q.q2, Case, kQ2);
    EXPECT_R_FIELD(q.q3, Case, kQ3);
    const auto f = statcpp::five_number_summary(x.begin(), x.end());
    EXPECT_R_FIELD(f.min, Case, kFnsMin);
    EXPECT_R_FIELD(f.q1, Case, kFnsQ1);
    EXPECT_R_FIELD(f.median, Case, kFnsMedian);
    EXPECT_R_FIELD(f.q3, Case, kFnsQ3);
    EXPECT_R_FIELD(f.max, Case, kFnsMax);
}

/// @brief Weighted median and weighted percentiles.
///
/// The last assertion records that with unit weights the definition coincides
/// with R's quantile(type = 2).
TEST(ROrderStatistics, Weighted) {
    using Case = ref::WeightedOrderStatistics;
    const auto x = AsVector(Case::kX);
    const auto w = AsVector(Case::kW);
    const std::vector<double> unit(x.size(), 1.0);
    EXPECT_R_FIELD(statcpp::weighted_median(x.begin(), x.end(), w.begin(), w.end()), Case,
                   kWeightedMedian);

    std::vector<double> actual;
    std::vector<double> unit_weighted;
    for (const double p : Case::kP) {
        actual.push_back(statcpp::weighted_percentile(x.begin(), x.end(), w.begin(), w.end(), p));
        unit_weighted.push_back(
            statcpp::weighted_percentile(x.begin(), x.end(), unit.begin(), unit.end(), p));
    }
    EXPECT_R_RANGE(actual, Case, kWeightedPercentile);
    EXPECT_R_RANGE(unit_weighted, Case, kUnitWeightCheck);
    EXPECT_R_RANGE(unit_weighted, Case, kQuantileType2);
}

// ============================================================================
// shape_of_distribution.hpp
// ============================================================================

/// @brief Skewness and kurtosis against e1071 types 1 and 2.
///
/// Both kurtosis forms are excess kurtosis. skewness and kurtosis are aliases
/// for the sample forms.
TEST(RShape, SkewnessAndKurtosis) {
    using Case = ref::Shape;
    const auto x = AsVector(Case::kX);
    const auto b = x.begin();
    const auto e = x.end();
    EXPECT_R_FIELD(statcpp::population_skewness(b, e), Case, kPopulationSkewness);
    EXPECT_R_FIELD(statcpp::sample_skewness(b, e), Case, kSampleSkewness);
    EXPECT_R_FIELD(statcpp::skewness(b, e), Case, kSkewness);
    EXPECT_R_FIELD(statcpp::population_kurtosis(b, e), Case, kPopulationKurtosis);
    EXPECT_R_FIELD(statcpp::sample_kurtosis(b, e), Case, kSampleKurtosis);
    EXPECT_R_FIELD(statcpp::kurtosis(b, e), Case, kKurtosis);
}

// ============================================================================
// correlation_covariance.hpp
// ============================================================================

/// @brief Correlation and covariance against cor and cov.
TEST(RCorrelation, CorrelationAndCovariance) {
    using Case = ref::Correlation;
    const auto x = AsVector(Case::kX);
    const auto y = AsVector(Case::kY);
    const auto xb = x.begin(), xe = x.end(), yb = y.begin(), ye = y.end();
    EXPECT_R_FIELD(statcpp::pearson_correlation(xb, xe, yb, ye), Case, kPearsonCorrelation);
    EXPECT_R_FIELD(statcpp::spearman_correlation(xb, xe, yb, ye), Case, kSpearmanCorrelation);
    EXPECT_R_FIELD(statcpp::kendall_tau(xb, xe, yb, ye), Case, kKendallTau);
    EXPECT_R_FIELD(statcpp::covariance(xb, xe, yb, ye), Case, kCovariance);
    EXPECT_R_FIELD(statcpp::sample_covariance(xb, xe, yb, ye), Case, kSampleCovariance);
    EXPECT_R_FIELD(statcpp::population_covariance(xb, xe, yb, ye), Case, kPopulationCovariance);
}

/// @brief Weighted covariance against cov.wt.
TEST(RCorrelation, WeightedCovariance) {
    using Case = ref::WeightedCovariance;
    const auto x = AsVector(Case::kX);
    const auto y = AsVector(Case::kY);
    const auto w = AsVector(Case::kW);
    EXPECT_R_FIELD(statcpp::weighted_covariance(x.begin(), x.end(), y.begin(), y.end(), w.begin()),
                   Case, kWeightedCovariance);
}

// ============================================================================
// distance_metrics.hpp
// ============================================================================

/// @brief Vector distances against R's dist metrics.
TEST(RDistance, Metrics) {
    using Case = ref::Distance;
    const auto x = AsVector(Case::kX);
    const auto y = AsVector(Case::kY);
    const auto xb = x.begin(), xe = x.end(), yb = y.begin(), ye = y.end();
    EXPECT_R_FIELD(statcpp::euclidean_distance(xb, xe, yb, ye), Case, kEuclideanDistance);
    EXPECT_R_FIELD(statcpp::manhattan_distance(xb, xe, yb, ye), Case, kManhattanDistance);
    // Chebyshev is R's "maximum" metric.
    EXPECT_R_FIELD(statcpp::chebyshev_distance(xb, xe, yb, ye), Case, kChebyshevDistance);
    EXPECT_R_FIELD(statcpp::minkowski_distance(xb, xe, yb, ye, Case::kP), Case,
                   kMinkowskiDistance);
    EXPECT_R_FIELD(statcpp::cosine_similarity(xb, xe, yb, ye), Case, kCosineSimilarity);
    EXPECT_R_FIELD(statcpp::cosine_similarity(xb, xe, yb, ye), Case, kProxyCosineCheck);
    EXPECT_R_FIELD(statcpp::cosine_distance(xb, xe, yb, ye), Case, kCosineDistance);
}

/// @brief Mahalanobis distance; R returns the squared value, statcpp the distance.
///
/// statcpp documents that only two dimensions are supported, so the reference is
/// two-dimensional and the rejection of a higher dimension is asserted as well.
TEST(RDistance, Mahalanobis) {
    using Case = ref::Mahalanobis;
    const auto x = AsVector(Case::kX);
    const auto mean = AsVector(Case::kMean);
    const auto cov = MakeMatrix(Case::kCovFlat, Case::kCovRows, Case::kCovCols);
    EXPECT_R_FIELD(statcpp::mahalanobis_distance(x, mean, cov), Case, kY);

    const std::vector<double> x3{2.5, 3.1, 4.7};
    const std::vector<double> mean3{2.0, 3.0, 4.0};
    const std::vector<std::vector<double>> cov3{
        {1.2, 0.3, 0.1}, {0.3, 0.9, 0.2}, {0.1, 0.2, 1.5}};
    EXPECT_THROW(statcpp::mahalanobis_distance(x3, mean3, cov3), std::invalid_argument);
}

// ============================================================================
// frequency_distribution.hpp
// ============================================================================

/// @brief Frequency, relative and cumulative tables against table() and prop.table().
TEST(RFrequency, Tables) {
    using Case = ref::Frequency;
    const auto x = AsVector(Case::kX);
    const auto b = x.begin();
    const auto e = x.end();

    // frequency_count returns an unordered map; sort by value to match table().
    const auto counts = statcpp::frequency_count(b, e);
    const std::map<double, std::size_t> ordered(counts.begin(), counts.end());
    std::vector<double> values, freq;
    for (const auto& [value, count] : ordered) {
        values.push_back(value);
        freq.push_back(static_cast<double>(count));
    }
    EXPECT_R_RANGE(values, Case, kValues);
    EXPECT_R_RANGE(freq, Case, kCount);

    const auto relative = statcpp::relative_frequency(b, e);
    const std::map<double, double> ordered_rel(relative.begin(), relative.end());
    std::vector<double> rel;
    for (const auto& [value, share] : ordered_rel) rel.push_back(share);
    EXPECT_R_RANGE(rel, Case, kRelative);

    std::vector<double> cum;
    for (const auto& [value, count] : statcpp::cumulative_frequency(b, e)) {
        cum.push_back(static_cast<double>(count));
    }
    EXPECT_R_RANGE(cum, Case, kCumulativeCount);

    std::vector<double> cum_rel;
    for (const auto& [value, share] : statcpp::cumulative_relative_frequency(b, e)) {
        cum_rel.push_back(share);
    }
    EXPECT_R_RANGE(cum_rel, Case, kCumulativeRelative);

    const auto table = statcpp::frequency_table(b, e);
    EXPECT_R_FIELD(static_cast<double>(table.total_count), Case, kTotal);
    std::vector<double> entry_values, entry_counts, entry_rel, entry_cum, entry_cum_rel;
    for (const auto& entry : table.entries) {
        entry_values.push_back(entry.value);
        entry_counts.push_back(static_cast<double>(entry.count));
        entry_rel.push_back(entry.relative_frequency);
        entry_cum.push_back(static_cast<double>(entry.cumulative_count));
        entry_cum_rel.push_back(entry.cumulative_relative_frequency);
    }
    EXPECT_R_RANGE(entry_values, Case, kValues);
    EXPECT_R_RANGE(entry_counts, Case, kCount);
    EXPECT_R_RANGE(entry_rel, Case, kRelative);
    EXPECT_R_RANGE(entry_cum, Case, kCumulativeCount);
    EXPECT_R_RANGE(entry_cum_rel, Case, kCumulativeRelative);
}

}  // namespace
