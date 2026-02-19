# Functions Not Verifiable Against R

The following functions cannot be automatically verified by numerical comparison with R, due to their nature. They are classified by reason.

---

## 1. Random / Stochastic Algorithms (Non-deterministic Output)

Results change on every execution, making comparison with specific R output values impossible.

### random_engine.hpp

| Function | Reason |
|----------|--------|
| `get_random_engine()` | Returns global random engine |
| `set_seed()` | Seed setting (independent of R's RNG) |

### resampling.hpp

| Function | Reason |
|----------|--------|
| `bootstrap_sample()` | Bootstrap sample (RNG-dependent) |
| `bootstrap()` | Bootstrap estimation (RNG-dependent) |
| `jackknife()` | Jackknife estimation (deterministic, but no generic R function) |
| `permutation_test()` | Permutation test (RNG-dependent) |
| `bootstrap_ci()` | Bootstrap confidence interval (RNG-dependent) |

### clustering.hpp

| Function | Reason |
|----------|--------|
| `kmeans()` | Initialization is RNG-dependent; convergence path differs from R |
| `kmeans_plusplus_init()` | k-means++ initialization (RNG-dependent) |

---

## 2. Utility / Data Manipulation (Not Statistical Computations)

Pure data manipulation functions that are not subject to comparison with R statistical functions.

### basic_statistics.hpp

| Function | Reason |
|----------|--------|
| `sum()` | Basic summation (trivial) |
| `count()` | Element count (trivial) |
| `argmin()` | Index of minimum value |
| `argmax()` | Index of maximum value |

### data_wrangling.hpp

| Function | Reason |
|----------|--------|
| `is_na()` | NA detection (NaN check) |
| `NA` (constant) | NA constant |
| `standardize()` | Z-standardization (indirectly covered by mean/sd verification) |
| `normalize()` | Min-max normalization |
| `rank_data()` | Ranking |
| `dummy_encode()` | Dummy encoding |
| `one_hot_encode()` | One-hot encoding |
| `remove_na()` | NA removal |
| `replace_na()` | NA replacement |
| `binning()` | Binning |

### missing_data.hpp

| Function | Reason |
|----------|--------|
| `complete_cases()` | Complete case extraction |
| `count_missing()` | Missing value count |
| `missing_rate()` | Missing rate |
| `mean_imputation()` | Mean imputation |
| `median_imputation()` | Median imputation |
| `mode_imputation()` | Mode imputation |
| `linear_interpolation()` | Linear interpolation |
| `listwise_deletion()` | Listwise deletion |
| `pairwise_deletion()` | Pairwise deletion |
| `hot_deck_imputation()` | Hot deck imputation (RNG-dependent) |
| `regression_imputation()` | Regression imputation |
| `multiple_imputation()` | Multiple imputation (RNG-dependent) |

---

## 3. Distance Metrics / Similarity (No Direct R Standard Function)

### distance_metrics.hpp

| Function | Reason |
|----------|--------|
| `euclidean_distance()` | Verifiable via `dist()`, but formula is trivial |
| `manhattan_distance()` | Same as above |
| `cosine_similarity()` | No standard R function |
| `cosine_distance()` | No standard R function |
| `chebyshev_distance()` | No standard R function |
| `minkowski_distance()` | Verifiable via `dist(method="minkowski")`, but formula is trivial |
| `hamming_distance()` | No standard R function |
| `jaccard_similarity()` | No standard R function |
| `jaccard_distance()` | No standard R function |
| `dice_coefficient()` | No standard R function |

---

## 4. Numerical Utilities (Not Subject to R Comparison)

### numerical_utils.hpp

| Function | Reason |
|----------|--------|
| `approx_equal()` | Floating-point comparison utility |
| `epsilon` (constant) | Machine epsilon |
| `default_rel_tol` (constant) | Default relative tolerance |
| `default_abs_tol` (constant) | Default absolute tolerance |

### special_functions.hpp

| Function | Reason |
|----------|--------|
| `lgamma()` | Available in R, but equivalent to C++ standard `std::lgamma` |
| `beta()` | Available in R, but formula is trivial |
| `log_beta()` | Same as above |
| `regularized_beta()` | No direct R counterpart (internal implementation) |
| `gamma_p()` / `gamma_q()` | Incomplete gamma functions (internal implementation) |
| `erf()` / `erfc()` | No direct R counterpart |
| Constants (`pi`, `sqrt_2`, etc.) | Mathematical constants (trivial) |

---

## 5. Variants of Basic Statistics (Indirectly Verified Through Representative Cases)

These are not directly tested in `verify_vs_r.cpp`, but are indirectly covered because verified functions use the same internal implementation.

### basic_statistics.hpp

| Function | Indirect Verification Source |
|----------|------------------------------|
| `mean()` | Used in t-test mean values, regression coefficient computation |
| `median()` | Used in MAD, Levene test |
| `mode()` / `modes()` | Not verified standalone |
| `geometric_mean()` | Not verified standalone |
| `harmonic_mean()` | Not verified standalone |
| `trimmed_mean()` | Not verified standalone |
| `weighted_mean()` | Not verified standalone |
| `logarithmic_mean()` | Not verified standalone |
| `weighted_harmonic_mean()` | Not verified standalone |

### dispersion_spread.hpp

| Function | Indirect Verification Source |
|----------|------------------------------|
| `variance()` / `sample_variance()` / `population_variance()` | Used in t-test, F-test p-value computation |
| `stddev()` / `sample_stddev()` / `population_stddev()` | Used in confidence intervals, regression SE |
| `iqr()` | Not verified standalone |
| `range()` | Not verified standalone |
| `coefficient_of_variation()` | Not verified standalone |
| `mean_absolute_deviation()` | Not verified standalone |
| `var()` / `stdev()` (ddof variants) | Not verified standalone |
| `weighted_variance()` / `weighted_stddev()` | Not verified standalone |
| `geometric_stddev()` | Not verified standalone |

### order_statistics.hpp

| Function | Indirect Verification Source |
|----------|------------------------------|
| `percentile()` | Used in Winsorize |
| `quartiles()` | Used in IQR |
| `five_number_summary()` | Not verified standalone |
| `decile()` | Not verified standalone |

### correlation_covariance.hpp

| Function | Indirect Verification Source |
|----------|------------------------------|
| `pearson_correlation()` | Indirectly verified through regression R-squared |
| `sample_covariance()` | Verified through covariance matrix |
| `spearman_correlation()` | Not verified standalone |
| `kendall_tau()` | Not verified standalone |
| `weighted_covariance()` | Not verified standalone |

### estimation.hpp

| Function | Indirect Verification Source |
|----------|------------------------------|
| `standard_error()` | Used indirectly in t-tests |
| `ci_proportion()` | Not verified standalone |
| `ci_variance()` | Not verified standalone |
| `ci_diff_means()` | Not verified standalone |
| `ci_diff_proportions()` | Not verified standalone |
| `proportion_test()` | Partially covered by z_test_proportion |

### frequency_distribution.hpp

| Function | Reason |
|----------|--------|
| `frequency_table()` | No directly comparable R output format |
| `histogram_bins()` | Bin calculation (differs from R's `hist` specification) |

---

## 6. Untested Overloads of Verified Functions

The following are Projection-argument overloads. The base versions have been verified, so these are not separately tested.

- `mean(first, last, proj)`, `median(first, last, proj)`, etc.
- `population_variance(first, last, proj)`, etc.
- `population_skewness(first, last, proj)`, etc.
- `pearson_correlation(first, last, first2, last2, proj1, proj2)`, etc.

---

## 7. Clustering (Reproducibility with R Not Guaranteed)

### clustering.hpp

| Function | Reason |
|----------|--------|
| `hierarchical_clustering()` | Comparable with R's `hclust()`, but tie-breaking can produce different results |
| `cut_dendrogram()` | Depends on hierarchical_clustering output |
| `silhouette_score()` | Label-dependent (kmeans is non-deterministic) |
| `euclidean_distance()` (clustering version) | Duplicates distance_metrics.hpp; formula is trivial |
| `manhattan_distance()` (clustering version) | Same as above |

---

## 8. Not Implemented (R Functions Missing from statcpp)

The following R features are not implemented in statcpp.

| R Function | Description |
|------------|-------------|
| `friedman.test()` | Friedman test |
| `p.adjust(method="holm")` | Holm correction |
| `p.adjust(method="hommel")` | Hommel correction |
| `p.adjust(method="hochberg")` | Hochberg correction |
| `p.adjust(method="BY")` | Benjamini-Yekutieli correction |

---

## Summary

| Category | Count | Numerical Checks |
|----------|-------|------------------|
| Directly verified against R | 57 functions | 167 checks (all PASS) |
| Indirectly verified (used internally) | ~30 functions | -- |
| Non-deterministic (RNG-dependent) | ~15 functions | -- |
| Utility / data manipulation | ~25 functions | -- |
| Distance metrics (trivial formulas) | ~10 functions | -- |
| Numerical utilities / special functions | ~10 functions | -- |
| Projection overloads | Many (base versions verified) | -- |
