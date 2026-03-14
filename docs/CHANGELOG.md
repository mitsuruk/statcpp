# Changelog

This document records the change history of the statcpp library.

This project follows [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-03-13

### Added

- **`nonparametric_tests.hpp` — `mann_whitney_u_test()`**: Added `bool correct = true` parameter for continuity correction (matches R's `wilcox.test` default).
- **`basic_statistics.hpp`, `dispersion_spread.hpp`, `order_statistics.hpp`**: Added new weight API overloads with explicit `WeightIterator` parameters. Old 3-argument overloads marked `[[deprecated]]`.
- **`basic_statistics.hpp`, `order_statistics.hpp`**: Added `static_assert` for random access iterator requirement.
- **`dispersion_spread.hpp` — `weighted_variance()` / `weighted_stddev()`**: Added new overloads accepting separate weight iterators with reliability weights semantics.

### Fixed

- **`robust.hpp` — `biweight_midvariance()`**: Fixed denominator weight function from `(1-u²)²` to `(1-u²)`.
- **`order_statistics.hpp` — `weighted_median()` / `weighted_percentile()`**: Skip zero-weight elements when cumulative weight reaches the boundary, finding the next positive-weight element.
- **`special_functions.hpp` — `erf()` / `erfc()`**: Replaced custom approximation (Abramowitz & Stegun, ~1.5e-7 precision) with `std::erf()` / `std::erfc()`.
- **`discrete_distributions.hpp` — `discrete_uniform_quantile()`**: Changed calculation from `floor(p * range)` to `ceil(p * range - 1)`.
- **`linear_regression.hpp` — `cook_distance()`**: Fixed denominator from `(1-h)` to `(1-h)²` (both simple and multiple regression).
- **`glm.hpp` — `glm_fit()`**: Preserved original `y_mean` before clipping for binomial/poisson null deviance calculation.
- **`clustering.hpp` — `kmeans()`**: Added fallback for `total_dist=0` in K-means++ initialization. Added farthest-point reinitialization for empty clusters.
- **`power_analysis.hpp` — `power_prop_test()`**: Rewrote to 2-stage method: `se_null` for rejection boundary, `se_alt` for power.
- **`data_wrangling.hpp` — `rank_transform()`**: Handle NaN values by assigning NaN ranks and computing ranks only for non-NaN elements.
- **`basic_statistics.hpp` — `weighted_harmonic_mean()`**: Unified near-zero detection with `harmonic_mean`.
- **`continuous_distributions.hpp` — `beta_pdf()` / `gamma_pdf()`**: Fixed boundary value handling.
- **`missing_data.hpp`**: Added `m >= 2` validation and row-length consistency checks.
- **`basic_statistics.hpp` — `mean()`**: Changed internal accumulation to `double` to prevent integer overflow.
- **`glm.hpp` — `glm_fit()`**: Fixed Gaussian AIC/BIC to count sigma² as estimated parameter.
- **`order_statistics.hpp` — `weighted_percentile()`**: Replaced exact floating-point comparison with tolerance-based comparison.
- **`resampling.hpp`**: Added `n_bootstrap < 2` validation and BCa index clamping.

### Changed

- **`nonparametric_tests.hpp` — `ks_test_normal()`**: Renamed to `lilliefors_test()`. Old name retained as `[[deprecated]]` alias.
- **`missing_data.hpp` — `test_mcar_simple()`**: Softened naming from "Little's MCAR test" to "Simple MCAR test (mean-difference based)".
- **`dispersion_spread.hpp` — `weighted_variance()`**: Documented as "reliability weights" (previously undocumented weight semantics).
- **Header guards**: Unified all headers to `#pragma once`.
- **`model_selection.hpp`**: Extracted `detail::standardize_features()` and `detail::rescale_coefficients()` helpers to reduce code duplication.
- **`estimation.hpp` — `ci_mean_diff_pooled()`**: Simplified to delegate to `ci_mean_diff()` (identical logic).

### Tests

- 847 unit tests with Google Test (758 at v0.1.0)
- 167 numerical verification checks against R 4.4.2
- Added `test_distance_metrics.cpp` (41 tests)
- Added erf/erfc NIST precision tests (5 tests)
- Added weighted variance/stddev tests (8 tests)

### Documentation

- Fixed `q.Q1`/`q.Q3` to `q.q1`/`q.q3` in sample code.
- Removed "Jackknife" and "Repeated Measures ANOVA" from feature lists where not applicable.
- Replaced `github.com/yourusername/statcpp` placeholder with `github.com/mitsuruk/statcpp`.
- Added `macOS + GCC 15 (Homebrew)` to verified environments.
- Translated `distance_metrics.hpp` comments to Japanese (JA version).
