# Changelog

This document records the change history of the statcpp library.

This project follows [Semantic Versioning](https://semver.org/).

## [0.1.2] - 2026-02-28

### Fixed

- **`basic_statistics.hpp` — `harmonic_mean`**: Changed zero-value detection from exact `val == 0.0` to `std::abs(val) < std::numeric_limits<double>::min()`.
  Subnormal values that would cause `1/val` to overflow to infinity are now also rejected.
  Error message updated to `"zero or near-zero value encountered"`.
- **`basic_statistics.hpp` — `logarithmic_mean`**: Replaced absolute-difference equality test `std::abs(x - y) < 1e-10` with relative-difference test `std::abs(x - y) <= 1e-10 * std::max(x, y)`.
  Prevents incorrect evaluation of `(y - x) / (ln y - ln x)` for very large but nearly-equal inputs (e.g., `x = 1e15`).
- **`order_statistics.hpp` — `weighted_median`**: Replaced exact `cumulative == half_weight` floating-point comparison with a tolerance-based check `std::abs(cumulative - half_weight) <= eps * half_weight`.
  Avoids missed boundary detection caused by floating-point accumulation rounding errors.
- **`correlation_covariance.hpp` — `weighted_covariance`**: Added clarifying comments that the Bessel correction formula `W / (W² − Σwᵢ²)` applies to **frequency weights**.
  For precision weights (inverse-variance), a different correction factor is required.
- **`correlation_covariance.hpp` — `kendall_tau`**: Added clarifying comments that tie detection uses exact equality (`diff == 0.0`), which is correct for integer or bit-identical double inputs.
  Callers passing computed floating-point values should round/quantise data beforehand.
- Added `<limits>` include to `basic_statistics.hpp` and `order_statistics.hpp` (both `include/` and `include-ja/`).

### Tests

- Added 18 new unit tests (758 → 776 total):
  - `HarmonicMeanTest.NearZeroValue` — subnormal value detection
  - `LogarithmicMeanTest.DistinctValues`, `.EqualValues`, `.LargeNearlyEqualValues`, `.NonPositiveArgument` — logarithmic mean coverage
  - `WeightedMedianTest.UniformWeights`, `.DominantWeight`, `.ExactHalfWeightBoundary`, `.EmptyRange`, `.NegativeWeight`, `.AllZeroWeights` — weighted median coverage
  - `KendallTauTest.PerfectPositive`, `.PerfectNegative`, `.TiedValues`, `.AllTiedInX`, `.EmptyRange` — Kendall's τ coverage
  - `WeightedCovarianceTest.UniformWeightsMatchSampleCovariance`, `.NegativeWeight` — weighted covariance coverage

---

## [0.1.1] - 2026-02-20

### Fixed

- Added 9 missing module headers to the umbrella header `statcpp.hpp`: `categorical.hpp`, `clustering.hpp`, `data_wrangling.hpp`, `missing_data.hpp`, `multivariate.hpp`, `power_analysis.hpp`, `robust.hpp`, `survival.hpp`, `time_series.hpp`
  - These modules were available as individual headers but were not included when using `#include <statcpp/statcpp.hpp>`
  - Both English (`include/`) and Japanese (`include-ja/`) umbrella headers have been updated

---

## [0.1.0] - 2025-02-19

### Initial Release

A comprehensive C++17 header-only statistics library with 758 unit tests and 167 R-verified numerical checks.

#### Features

##### Basic Statistics (`basic_statistics.hpp`)

- Sum, mean, median, mode
- Geometric mean, harmonic mean, trimmed mean
- Weighted mean, logarithmic mean
- Projection overloads for all functions

##### Dispersion & Spread (`dispersion_spread.hpp`)

- Variance (population/sample), standard deviation, range
- Coefficient of variation, interquartile range
- Mean absolute deviation
- Pre-computed mean overloads

##### Order Statistics (`order_statistics.hpp`)

- Minimum, maximum
- Quartiles, percentiles (R type=7 / Excel PERCENTILE.INC compatible)
- Five-number summary

##### Shape of Distribution (`shape_of_distribution.hpp`)

- Skewness
- Kurtosis, excess kurtosis

##### Correlation & Covariance (`correlation_covariance.hpp`)

- Pearson correlation coefficient
- Spearman rank correlation coefficient
- Kendall's rank correlation coefficient
- Covariance

##### Frequency Distribution (`frequency_distribution.hpp`)

- Frequency distribution table
- Histogram
- Cumulative frequency distribution

##### Special Functions (`special_functions.hpp`)

- Gamma function, beta function, log-gamma
- Error function
- Incomplete beta/gamma functions and their inverses

##### Random Number Generation (`random_engine.hpp`)

- Thread-safe default random engine
- Seeding utilities

##### Probability Distributions

- Continuous distributions (`continuous_distributions.hpp`): Normal, t, chi-squared, F, exponential, gamma, beta, Weibull, log-normal, Cauchy, studentized range
  - CDF, PDF, quantile, and random generation for each
  - Studentized range CDF via Copenhaver & Holland (1988) algorithm (matching R's `ptukey`)
  - Newton-Raphson quantile functions with `isfinite` guards for numerical robustness
- Discrete distributions (`discrete_distributions.hpp`): Binomial, Poisson, geometric, negative binomial, hypergeometric

##### Statistical Estimation (`estimation.hpp`)

- Confidence intervals for mean, proportion, variance

##### Hypothesis Tests

- Parametric tests (`parametric_tests.hpp`): z-test, t-test (one/two sample, paired), F-test, chi-squared test
- Nonparametric tests (`nonparametric_tests.hpp`): Wilcoxon signed-rank test, Mann-Whitney U test, Kruskal-Wallis test, Friedman test, Kolmogorov-Smirnov test, Levene's test, Bartlett's test
  - Tie corrections for Wilcoxon, Mann-Whitney U, and Kruskal-Wallis tests

##### Effect Size (`effect_size.hpp`)

- Cohen's d, Hedges' g, Glass's delta
- Eta squared, omega squared
- Cohen's f, R-squared

##### Resampling (`resampling.hpp`)

- Bootstrap (percentile, BCa)
- Jackknife
- Permutation test

##### Power Analysis (`power_analysis.hpp`)

- Sample size calculation and power for t-test (one/two sample)
- Sample size calculation and power for proportion test
- Both string-based and `alternative_hypothesis` enum overloads

##### Regression Analysis

- Simple and multiple linear regression (`linear_regression.hpp`)
- Generalized linear models (`glm.hpp`): Gaussian, binomial, Poisson, gamma families with identity, logit, probit, log, inverse, cloglog link functions
  - IRLS algorithm with convergence tracking and NaN-safe output

##### ANOVA (`anova.hpp`)

- One-way ANOVA with degenerate case handling
- Two-way ANOVA
- Repeated measures ANOVA
- Post-hoc tests: Tukey HSD (true studentized range), Bonferroni, Dunnett, Scheffe
  - Degenerate `se == 0` handling for all post-hoc functions

##### Model Selection (`model_selection.hpp`)

- AIC, BIC, adjusted R-squared
- Cross-validation
- Stepwise selection

##### Multivariate Analysis (`multivariate.hpp`)

- PCA (principal component analysis)
- Basic multivariate functions

##### Clustering (`clustering.hpp`)

- k-means clustering
- Hierarchical clustering (single, complete, average, Ward's linkage)

##### Time Series (`time_series.hpp`)

- ACF, PACF
- Moving average, exponential smoothing
- Basic time series operations

##### Categorical Data Analysis (`categorical.hpp`)

- Chi-squared test of independence
- Fisher's exact test

##### Survival Analysis (`survival.hpp`)

- Kaplan-Meier estimator
- Log-rank test

##### Robust Statistics (`robust.hpp`)

- Median absolute deviation (MAD)
- Trimmed/Winsorized statistics

##### Data Wrangling (`data_wrangling.hpp`, `missing_data.hpp`)

- NaN removal, data filtering
- Missing data handling

##### Distance & Similarity (`distance_metrics.hpp`)

- Euclidean, Manhattan, Chebyshev, Minkowski distances
- Cosine similarity, Jaccard coefficient, Hamming distance

##### Numerical Utilities (`numerical_utils.hpp`)

- Approximate equality for floating-point numbers
- Kahan summation algorithm
- Numerically stable logarithm and exponential functions
- Value clamping and range checking

#### Architecture

- C++17 header-only library (no build required)
- Iterator-based interface compatible with STL containers
- Headers organized under `include/statcpp/` with `#include "statcpp/module.hpp"` convention
- Bilingual headers: English (`include/`) and Japanese (`include-ja/`)
- CMake support with `find_package(statcpp)` and `add_subdirectory`
- Cross-platform: macOS (Apple Clang), Linux (GCC), Windows (MSVC)
- MIT License

#### Documentation

- Complete Doxygen API documentation
- Usage guide with include path setup
- Sample programs for each module (30+)
- Installation, building, and contributing guides

#### Testing

- 758 unit tests with Google Test
- 167 numerical verification checks against R 4.4.2
- Edge case and exception coverage

---

## Changelog Format

Future releases will record changes in the following format:

### Added

New features

### Changed

Changes to existing features

### Deprecated

Features scheduled for removal

### Removed

Removed features

### Fixed

Bug fixes

### Security

Security-related changes

---

[0.1.1]: https://github.com/yourusername/statcpp/releases/tag/v0.1.1
[0.1.0]: https://github.com/yourusername/statcpp/releases/tag/v0.1.0
