# NaN (Missing Value) Policy

> **Status: implemented in v0.5.0 (2026-10-07)**

This document defines how every public function of statcpp behaves when its input
contains NaN. Before v0.4.0 there was no such policy and the behaviour varied from
function to function (section 6, "Problems in v0.4.0"). v0.5.0 brings every public
function in line with this policy (sections 9 and 10).

---

## 1. Terms

- **NaN**: `std::numeric_limits<double>::quiet_NaN()`. statcpp represents a missing
  value (NA) with it, the same value as the `statcpp::NA` constant. The kind of NaN
  (quiet or signaling, sign) is not distinguished.
- **±Inf**: **not** a missing value. It is treated as an ordinary value (outside the
  scope of this policy).
- **Data argument**: a sequence of observations (iterator range, `std::vector`,
  matrix). The weights of weighted statistics are included.
- **Parameter argument**: any numeric argument other than observations (`mu`,
  `sigma`, `alpha`, confidence level, window size and so on).
- **Evaluation point**: the point at which a distribution function or similar is
  evaluated (`x` in `normal_pdf(x, ...)`, `p` in `normal_quantile(p, ...)`).

---

## 2. Principles

1. **Follow R**: each function follows the NA handling of its R counterpart when
   that counterpart is called **with its default arguments**.

   | R behaviour | statcpp behaviour |
   | --- | --- |
   | returns `NA` / `NaN` | returns NaN |
   | removes NA, then computes | removes NaN, then computes |
   | raises an error | throws `std::invalid_argument` |

2. **When R has no counterpart**, the class default of section 4 applies.
3. **Where statcpp deliberately differs from R**, the difference and its reason are
   listed in section 5 and recorded in `testWithR/METHODOLOGY.md`.
4. **Never fail silently**: no function may loop forever, invoke undefined behaviour
   or return a result that depends on the position of NaN when its input contains
   NaN. This is the minimum requirement, before R compliance.

Adding NA cases to the R verification tests checks compliance with this policy
mechanically.

---

## 3. R default behaviour (measured with R 4.4.2)

Measured with inputs such as `c(3, 1, NA, 4, 5, 2)`. The corresponding statcpp
functions follow these results.

| Area | R behaviour | Examples of R functions |
| --- | --- | --- |
| Descriptive statistics | returns NA | `mean`, `var`, `sd`, `min`, `max`, `range`, `median`, `mad`, `cor` (pearson / spearman) |
| Descriptive statistics | error | `quantile`, `IQR` |
| Descriptive statistics | removes NA | `fivenum` |
| Element-wise output | NA only in the affected element | `diff`, `log` and other element-wise transforms |
| Cumulative output | NA from the NA onwards | `cumsum` |
| Sorting | removes NA | `sort` |
| Sorting | places NA last | `order` |
| Inference | removes NA (pairwise for paired two-sample tests) | `t.test`, `wilcox.test`, `shapiro.test`, `ks.test`, `var.test`, `cor.test`, `kruskal.test`, `aov`, `lm` |
| Inference | error | `chisq.test`, `kmeans`, `prcomp` |
| Mathematical functions | returns NaN | `dnorm(NaN)`, `qnorm(NaN)`, `lgamma(NaN)` |
| Parameters | returns NaN (with a warning) | `dnorm(0, 0, NaN)`, `rnorm(1, 0, NaN)`, `dpois(2, NaN)` |

The per-function mapping (statcpp function, R counterpart and its default
behaviour, behaviour under this policy) is in the appendix
[NAN_INVENTORY.md](NAN_INVENTORY.md).

---

## 4. Class defaults when R has no counterpart

| Class | Scope | Default |
| --- | --- | --- |
| C1 scalar statistics | descriptive statistics, distances and other functions returning one real number | returns NaN (as most of R's descriptive statistics do) |
| C2 element-wise and window-wise output | rolling functions, transforms, lags and differences | NaN only in the elements or windows that depend on a NaN. Cumulative functions (`exponential_moving_average` and so on) become NaN from the NaN onwards, as R's `cumsum` does |
| C3 integer and categorical output | bin numbers, class codes, counts | throws `std::invalid_argument` (section 5) |
| C6 inference and model estimation | tests, estimators and models returning a struct | removes NaN, then computes (as most of R's tests do). Paired two-sample functions remove incomplete pairs |
| C7 mathematical functions | distribution and special functions | returns NaN when the evaluation point or a parameter is NaN |

---

## 5. Deliberate differences from R

| Scope | R | statcpp | Reason |
| --- | --- | --- | --- |
| Functions returning integers or enums (`bin_equal_width`, `bin_equal_freq`, `label_encode`, `one_hot_encode`, `value_counts`, `frequency_*`, discrete `*_quantile` / `*_rand`, `sample_size_*`, `interpret_*`, the outlier detectors `detect_outliers_*`) | returns NA, or excludes NA | throws `std::invalid_argument` | an integer or enum return type (per-point outlier flags for the outlier detectors) cannot represent NA, and changing return types would break the API |
| A NaN range-checked parameter (confidence level, `alpha`, quantile `p`, trim proportion, `lambda`, `tol`, window size, sampling ratio and so on) | NA or an error, depending on the function | throws `std::invalid_argument` | treated as an invalid parameter, like an out-of-range value. Parameters of distribution functions are excluded (they return NaN, as in R) |
| A NaN bound in `filter_range` or `validate_range` | no counterpart | throws `std::invalid_argument` | a NaN bound makes the check pass silently (v0.4.0 always returned true) |
| `fillna_ffill`, `fillna_bfill`, `fillna_interpolate` | `zoo::na.locf` / `zoo::na.approx` drop leading or trailing NA by default, shortening the result | keep the length and leave a NaN that cannot be filled as NaN | keeps the contract of returning a vector aligned with the input |
| `rank_transform`, `compute_ranks_with_ties` | `rank` gives NA the largest rank by default (`na.last = TRUE`) | keeps NaN as NaN (equivalent to `rank(na.last = "keep")`) | a NaN ranked like an ordinary value is easily misused downstream |
| Clustering (`hierarchical_clustering`, `silhouette_score`) | `kmeans` errors; functions built on `dist()` skip NA coordinates | throws `std::invalid_argument` | handles every clustering function the same way as `kmeans` |
| `two_way_anova` | computes unbalanced designs too | removes NaN, then applies the existing balanced-design check (unbalanced throws) | statcpp's `two_way_anova` assumes a balanced design |
| A finite but invalid parameter (`sigma <= 0` and so on) | returns NaN with a warning | throws `std::invalid_argument`, as before | existing behaviour; this policy changes only the handling of NaN |

---

## 6. Problems in v0.4.0 (fixed in v0.5.0)

All 391 functions (counted per module, duplicates across headers included) were
measured; the results are in the appendix [NAN_INVENTORY.md](NAN_INVENTORY.md).
Each function received NaN in two positions, "middle" and "first", and ran in a
separate process with a 10-second limit. The runs were repeated with
AddressSanitizer and UndefinedBehaviorSanitizer and compared against R 4.4.2 with
default arguments.

| Verdict | Count | Meaning |
| --- | --- | --- |
| OK | 200 | already complied with this policy |
| FIX | 153 | differed from this policy (mostly by silently returning a wrong value) |
| FIX-UB | 22 | undefined behaviour (NaN cast to an integer, `std::sort` over comparisons involving NaN) |
| FIX-HANG | 16 | infinite loop |

The root causes reduce to these patterns.

| Root cause | Representative functions |
| --- | --- |
| A loop grouping equal values never advances on NaN (`NaN == NaN` is false) | `spearman_correlation`, `mann_whitney_u_test`, `wilcoxon_signed_rank_test`, `kruskal_wallis_test`, `kaplan_meier`, `nelson_aalen` (infinite loop) |
| A NaN parameter passed to a standard library distribution | `gamma_rand`, `poisson_rand` and 6 other functions (infinite loop) |
| `std::sort` / `std::min_element` over NaN | `minimum`, `maximum`, `range`, `mad`, `weighted_median`, `holm_correction`, `sort_values` |
| NaN cast to an integer | `percentile`, `trimmed_mean`, the `bootstrap` family, `bin_equal_width`, `discrete_uniform_quantile` |
| A NaN parameter slips through a check of the form `x <= 0` | estimators, tests and power functions taking a confidence level or `alpha` |
| NaN used as a `std::map<double>` key | `value_counts`, `group_*`, `frequency_*`, `label_encode` |
| NaN not removed (R removes it) | most inferential functions, such as `t_test` and `ci_mean` |

Examples that silently returned a wrong conclusion:

| Function | v0.4.0 |
| --- | --- |
| `permutation_test_two_sample` and 2 others | one NaN gave p = 0.002 (p = 0.85 without the NaN), a false positive |
| `chisq_test_independence` | a NaN cell gave χ² = 0, p = 1 |
| `shapiro_wilk_test` | W = 1, p = 1 (wrongly concluding normality) |
| `lilliefors_test`, `ks_test_normal` | D = 0, p = 1 |
| GLM functions | `converged = true` with NaN coefficients |
| `detect_outliers_iqr` | one NaN made every point an outlier |
| `tukey_hsd` | p = 2.4e-14 |
| `percentile` family | a finite value that depended on the position of NaN |

A bug unrelated to NaN was also found: `fillna_median` called `median`, which
expects sorted input, on unsorted data and filled with the wrong median (for
`{9, NaN, 1, 5, 3}` it filled 3 instead of 4). It is fixed in v0.5.0.

---

## 7. Implementation rules

- Detecting, removing and ordering NaN last are done by shared helpers in
  `statcpp::detail` (`nan_utils.hpp`) rather than repeated in every function.
- Exception messages follow the existing form `"statcpp::<function>: <reason>"`
  (for example `"statcpp::bin_equal_width: data contains NaN"`).
- A function that removes NaN applies its existing checks (too few observations
  and so on) to the data after removal.
- Functions comparable with R get NA cases in the R verification tests; the others
  are checked against their class default by unit tests.

---

## 8. Out of scope

- The handling of ±Inf (kept as ordinary values). The one exception is
  `moving_average`, whose running sum turned Inf - Inf into NaN and made every
  later window NaN; v0.5.0 computes it per window, as R's `stats::filter` does.
- The handling of finite but invalid parameters (unchanged, as stated in section 5).
- Distinguishing kinds of NaN (signaling or quiet, sign, payload).

---

## 9. Decisions made during implementation

Functions that sections 4 and 5 did not settle were implemented as follows. Where R
has a counterpart, its default behaviour is followed.

| Scope | v0.5.0 behaviour | Basis |
| --- | --- | --- |
| `sort_values` / `argsort` | `sort_values` removes NaN; `argsort` places NaN last in its original order | R's `sort` / `order(na.last = TRUE)` |
| `argmin` / `argmax` | the position among the non-NaN values; all NaN throws | R's `which.min` / `which.max` |
| `drop_duplicates` / `get_duplicates` | NaN values are equal to each other | R's `unique` / `duplicated` |
| `group_by`, `group_mean`, `group_sum`, `group_count` | rows with a NaN key are dropped | R's `split` / `tapply` make no NA group |
| `rolling_min` / `rolling_max` | a window containing NaN is NaN wherever the NaN is | class default C2 |
| `clamp` | NaN if any argument is NaN | R's `pmin(pmax(x, lo), hi)` |
| `euclidean_distance`, `manhattan_distance`, `minkowski_distance` (both headers) | NaN coordinates are skipped and the sum is scaled by n / n_used; NaN when no coordinate is usable | R's `dist()` |
| `cosine_similarity` / `cosine_distance` | pairs containing NaN are dropped; NaN when no pair is usable | `proxy::simil` (no base R counterpart) |
| `kendall_tau`, `spearman_correlation` | returns NaN | R's `cor(method = ...)` default (`use = "everything"`) |
| `weighted_covariance`, `weighted_variance`, `weighted_stddev` | throws | R's `cov.wt` errors |
| `autocorrelation`, `acf`, `pacf` | throws | R's `acf` default `na.action = na.fail` |
| `ci_mean`, `ci_mean_z`, `ci_variance`, `margin_of_error_mean`, `ci_mean_diff*` | NaN removed, then computed | consistent with R's `t.test` (class default C6) |
| Regression, GLM and cross-validation (`simple_linear_regression`, `glm_fit` and others) | rows containing NaN are dropped; residuals and similar vectors have the reduced length | R's `lm` / `glm` with `na.omit` |
| `pseudo_r_squared_nagelkerke` | NaN responses are removed and n is reduced by their number | consistent with a model fitted with `na.omit` |
| Regularized regression, `cv_ridge`, `cv_lasso`, `generate_lambda_grid` | throws; a NaN entry in the `lambda` grid also throws | `glmnet` errors on missing values |
| `correlation_matrix` | NaN propagates element-wise and the diagonal is 1 | R's `cor` |
| `standardize` | column statistics from the non-NaN values; only the NaN cell is NaN | R's `scale` |
| `power_iteration`, `pca`, `pca_transform`, clustering | throws | R's `prcomp` / `kmeans` error |
| `permutation_test_*` | NaN removed; the two paired functions remove incomplete pairs | as R's `t.test` / `cor.test`. v0.4.0 returned a false positive (p = 1/(B+1)) |
| Generic `bootstrap` / `bootstrap_bca` | data passed to the statistic unchanged; a NaN confidence level throws | R's `boot` |
| `bootstrap_mean` / `bootstrap_median` / `bootstrap_stddev` | NaN removed | class default C6 |
| Continuous `*_rand` | returns NaN when a parameter is NaN | R's `rnorm(1, 0, NaN)` and so on |
| `multiple_imputation_bootstrap` | a column with no observed value in the bootstrap sample is imputed from the observed values of the original data; an entirely missing column stays NaN | v0.4.0 silently imputed 0 |
| Reference values of tests such as `mu0` | NaN throws | R's `t.test(mu = NA)` errors |
| `alpha` of post-hoc tests | NaN throws | a range-checked parameter (section 5) |

---

## 10. Verification

- **Unit tests**: each module's tests have a "NaN Handling (v0.5.0)" section. Every
  test was confirmed to fail before its fix. Including the NaN helper tests
  (`test_nan_utils.cpp`), `statcpp_tests` has 974 tests.
- **R verification**: `testWithR/test_vs_r_nan.cpp` compares 23 cases against the
  default results of R 4.4.2 on inputs containing NA (removal, propagation,
  element-wise, rescaled distances, row removal, ordering). Run against the v0.4.0
  headers, all 23 cases fail or stop.
- **Configurations**: all 1161 tests pass in six configurations (Debug, Release,
  RelWithDebInfo, Japanese headers, libc++ hardening, AddressSanitizer /
  UndefinedBehaviorSanitizer), with no UBSan reports.
