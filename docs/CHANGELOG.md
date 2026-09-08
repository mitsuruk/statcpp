# Changelog

This document records the change history of the statcpp library.

This project follows [Semantic Versioning](https://semver.org/).

## [0.4.0] - 2026-09-08

Numerical corrections found by cross-verifying the whole library against R 4.4.2.
Every public signature from 0.3.0 still compiles: the only API changes are one new
function and trailing defaulted parameters. **The behaviour of `lilliefors_test()`
changes substantially**, so downstream results will differ; see the upgrade note
below.

### Fixed

- **`nonparametric_tests.hpp` — `lilliefors_test()` / `ks_test_normal()`**: The p-value now uses the Dallal & Wilkinson (1986) analytic approximation. The previous `p = 2·exp(-2·d_adj²)` tail formula was applied over the whole range and disagreed with `nortest::lillie.test` everywhere: for a near-normal sample it returned 0.0295 where R returns 0.729, **falsely rejecting normality at α = 0.05**. The new form reproduces R exactly for p ≤ 0.05 (723 of 723 samples) and produced no disagreement in the reject/accept decision at α = 0.05 or 0.01 over 2,400 samples. Above p = 0.10 the approximation is outside its published range and only indicates consistency with normality. The D statistic is unchanged.
- **`special_functions.hpp` — `norm_cdf()`**: Evaluated as `0.5·erfc(-x/√2)` instead of `0.5·(1 + erf(x/√2))`. The old form cancelled catastrophically in the left tail: it was already 40% wrong at x = -8.25 and returned exactly zero from x = -8.327, where the true value is still far inside the representable range. The new form agrees with R to 1.9e-13 relative down to x = -37.5, and also improves the centre from 2.2e-14 to 1.7e-15. `normal_cdf()` and `lognormal_cdf()` inherit the fix.
- **`parametric_tests.hpp`, `nonparametric_tests.hpp`, `glm.hpp`, `power_analysis.hpp` — upper-tail p-values**: Nineteen sites that formed a p-value as `1 - norm_cdf(z)` now use `norm_sf(z)`. The old form returned exactly zero for `|z| ≥ 8.30`. A Poisson regression intercept p-value went from 0 to 1.2688719e-117 against R's 1.2688719e-117. Affects `z_test()`, `z_test_proportion()`, `z_test_proportion_two_sample()`, `mann_whitney_u_test()`, `wilcoxon_signed_rank_test()`, GLM Wald p-values and the power functions.

### Added

- **`special_functions.hpp` — `norm_sf()`**: Standard normal survival function `P(Z > x)`, evaluated through `erfc`. Use it instead of `1 - norm_cdf(x)` whenever an upper tail probability is needed.
- **`model_selection.hpp` — `cross_validate_linear()` / `cv_ridge()` / `cv_lasso()`**: Added a trailing `bool shuffle = true` parameter, forwarded to `create_cv_folds()`. Passing `false` produces contiguous, deterministic folds, making cross-validation reproducible. The default preserves 0.3.0 behaviour.

### Changed

- **`testWithR/`**: The standalone `verify_vs_r` program (167 hand-transcribed reference values covering 57 functions) has been removed and replaced by a Google Test suite whose reference values are generated from R. It covers **all 321 R-comparable public functions** in 164 tests, comparing 48,598 values. R runs as a separate process at generation time and is not linked, so statcpp and the test binary remain MIT licensed.
- **`testWithR/VERIFIED_FUNCTIONS.md`, `testWithR/NON_VERIFIABLE_FUNCTIONS.md`**: Replaced by `R_VERIFICATION_INVENTORY.ja.md` (full classification of all public functions) and `VERIFICATION_CHECKLIST.ja.md` (per-function progress).

### Documentation

- Corrected the public function count in `README.md`, `README.ja.md` and both `API_REFERENCE.md` files: 386 unique names, 538 including overloads. The previous figure of 524 did not match the code at any commit, including the one that introduced it.
- Corrected the test count from 793 to 857 unit tests plus 164 verification tests.
- Added `norm_sf` to both API references, and documented `betainc_impl` / `lgamma_impl` as implementation helpers that are not part of the intended public interface.
- Removed an incorrect reference to a Cauchy distribution, which the library does not provide (English API reference only).
- Rewrote `testWithR/METHODOLOGY.md`, which now records 22 measured definition differences against R, including the quantile type, `mad()` versus mean absolute deviation, `fivenum()` versus type 7 quantiles, weighted variance semantics, and the intercept handling of `odds_ratios()`.

### Upgrade notes

- **Source compatibility**: nothing was removed or renamed. All 302 `statcpp::` symbols referenced by the known downstream projects still exist.
- **Result compatibility**: `lilliefors_test()` changes its conclusion for a large fraction of inputs. Over 3,000 random samples the reject/accept decision flipped for 61.6% at α = 0.05, and in every case the 0.3.0 result was a false rejection. Golden files and cross-validation tolerances that pinned the old p-value need regenerating.
- Everything else changes only in the far tail, where 0.3.0 returned exactly zero. `power_t_test_one_sample()`, `power_t_test_two_sample()` and `power_prop_test()` change by at most 2e-15, and the five `sample_size_*` functions returned identical integers across 3,653 checked cases.

## [0.3.0] - 2026-07-09

Correctness and boundary-safety fixes from a full-library computation review. All
public signatures are unchanged, but several functions now return corrected values.

### Fixed

- **`linear_regression.hpp` — `compute_residual_diagnostics()`**: Cook's distance denominator corrected from `(1-h)` to `(1-h)²` in the English header (the Japanese header was already correct in 0.2.0; this resolves an `include/` vs `include-ja/` drift). High-leverage points were under-reported by up to ~6x.
- **`discrete_distributions.hpp` — `poisson_quantile()` / `geometric_quantile()` / `nbinom_quantile()`**: Added `prob == 1.0` guard. Previously cast `+inf` to `uint64` (undefined behavior); now returns `std::numeric_limits<uint64_t>::max()` for these infinite-support distributions.
- **`continuous_distributions.hpp` — `beta_rand()`**: Re-draw when both gamma variates underflow to 0 (extremely small shapes, e.g. α=β=0.001), which previously produced silent NaN.
- **`data_wrangling.hpp` — `rolling_mean()` / `rolling_sum()`**: A NaN now contaminates only the windows that actually contain it. The incremental sum previously stayed NaN for all subsequent windows. Now consistent with `rolling_std/min/max`.
- **`glm.hpp` — `glm_fit()` coefficient SE**: Apply the dispersion φ to the coefficient covariance (`φ·(XᵀWX)⁻¹`). φ is fixed at 1 for binomial/poisson and estimated by the Pearson statistic / residual df for gaussian/gamma (as in R's `summary.glm`). Gaussian SEs now match OLS; z-statistics and p-values follow.
- **`glm.hpp` — Poisson null log-likelihood**: Added the missing `-lgamma(y+1)` term so McFadden's pseudo-R² is consistent between the null and fitted models.
- **`glm.hpp` — gamma log-likelihood**: Corrected to the exact gamma(shape=ν, mean=μ) density `ν·log(ν/μ) − logΓ(ν) + (ν−1)·log(y) − ν·y/μ` (previously missing the `ν·log ν` term and using a `2ν−2` coefficient on `log y`). Affects gamma AIC/BIC.
- **`nonparametric_tests.hpp` — `kruskal_wallis_test()`**: All-tied input previously divided by a zero tie-correction factor, giving `-inf`/NaN. Now returns H=0, p=1.
- **`nonparametric_tests.hpp` — `shapiro_wilk_test()`**: For W ≥ 1 (maximally normal, e.g. perfectly equispaced data) the p-value was ~0.001 (strongly rejecting normality); it is now ~1.

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
