# R Verification Checklist

Progress tracker for the **321 public functions** of statcpp that are compared
numerically against R 4.4.2.

See `R_VERIFICATION_INVENTORY.md` for how the set was chosen and which R
function each one is compared with.

## Summary

**All five phases are complete: every one of the 321 comparable functions has been
checked against R 4.4.2.**

All 164 tests in `statcpp_r_tests` pass.

Installing `glmnet`, `mice` and `naniar`, and adding a `shuffle` argument to the
cross-validation functions, brought eight functions into scope that had been
classified as conditional or fold-dependent (313 to 321).

Fixing the upper-tail cancellation in p-values formed from z statistics added the
survival function `norm_sf` as a public function (385 to 386 public functions,
312 to 313 comparable).

`cross_validate_linear` turned out to reach the global random engine through
`create_cv_folds(shuffle = true)`, so it was first removed from scope and the
total corrected from 313 to 312, then restored once `shuffle` was added.

## Assumptions

- Verification runs on macOS only
- R runs as a separate `Rscript` process and is never linked, which keeps the GPL
  out of the MIT-licensed library and test binary
- Scope: 390 public entries (385 unique names) minus 70 not comparable, plus
  `norm_sf`, giving **321**
- The 24 functions in `statcpp::detail` are internal and out of scope

## Progress

|Header|Target|Done|
|---|---|---|
|`parametric_tests.hpp`|14|14|
|`nonparametric_tests.hpp`|10|10|
|`anova.hpp`|13|13|
|`continuous_distributions.hpp`|32|32|
|`discrete_distributions.hpp`|24|24|
|`special_functions.hpp`|14|14|
|`basic_statistics.hpp`|13|13|
|`dispersion_spread.hpp`|15|15|
|`order_statistics.hpp`|7|7|
|`shape_of_distribution.hpp`|6|6|
|`correlation_covariance.hpp`|7|7|
|`distance_metrics.hpp`|7|7|
|`linear_regression.hpp`|11|11|
|`glm.hpp`|12|12|
|`model_selection.hpp`|14|14|
|`effect_size.hpp`|15|15|
|`estimation.hpp`|15|15|
|`power_analysis.hpp`|8|8|
|`survival.hpp`|4|4|
|`time_series.hpp`|12|12|
|`robust.hpp`|10|10|
|`multivariate.hpp`|7|7|
|`categorical.hpp`|5|5|
|`frequency_distribution.hpp`|5|5|
|`clustering.hpp`|5|5|
|`data_wrangling.hpp`|27|27|
|`missing_data.hpp`|6|6|
|`numerical_utils.hpp`|3|3|
|**Total**|**321**|**321**|

---

## Checklist

### parametric_tests.hpp (14)

- [x] `benjamini_hochberg_correction`
- [x] `bonferroni_correction`
- [x] `chisq_test_gof`
- [x] `chisq_test_gof_uniform`
- [x] `chisq_test_independence`
- [x] `f_test`
- [x] `holm_correction`
- [x] `t_test`
- [x] `t_test_paired`
- [x] `t_test_two_sample`
- [x] `t_test_welch`
- [x] `z_test`
- [x] `z_test_proportion`
- [x] `z_test_proportion_two_sample`

### nonparametric_tests.hpp (10)

- [x] `bartlett_test`
- [x] `compute_ranks_with_ties`
- [x] `fisher_exact_test`
- [x] `kruskal_wallis_test`
- [x] `ks_test_normal`
- [x] `levene_test`
- [x] `lilliefors_test`
- [x] `mann_whitney_u_test`
- [x] `shapiro_wilk_test`
- [x] `wilcoxon_signed_rank_test`

### anova.hpp (13)

- [x] `bonferroni_posthoc`
- [x] `cohens_f`
- [x] `dunnett_posthoc`
- [x] `eta_squared`
- [x] `omega_squared`
- [x] `one_way_ancova`
- [x] `one_way_anova`
- [x] `partial_eta_squared_a`
- [x] `partial_eta_squared_b`
- [x] `partial_eta_squared_interaction`
- [x] `scheffe_posthoc`
- [x] `tukey_hsd`
- [x] `two_way_anova`

### continuous_distributions.hpp (32)

- [x] `beta_cdf`
- [x] `beta_pdf`
- [x] `beta_quantile`
- [x] `chisq_cdf`
- [x] `chisq_pdf`
- [x] `chisq_quantile`
- [x] `exponential_cdf`
- [x] `exponential_pdf`
- [x] `exponential_quantile`
- [x] `f_cdf`
- [x] `f_pdf`
- [x] `f_quantile`
- [x] `gamma_cdf`
- [x] `gamma_pdf`
- [x] `gamma_quantile`
- [x] `lognormal_cdf`
- [x] `lognormal_pdf`
- [x] `lognormal_quantile`
- [x] `normal_cdf`
- [x] `normal_pdf`
- [x] `normal_quantile`
- [x] `studentized_range_cdf`
- [x] `studentized_range_quantile`
- [x] `t_cdf`
- [x] `t_pdf`
- [x] `t_quantile`
- [x] `uniform_cdf`
- [x] `uniform_pdf`
- [x] `uniform_quantile`
- [x] `weibull_cdf`
- [x] `weibull_pdf`
- [x] `weibull_quantile`

### discrete_distributions.hpp (24)

- [x] `bernoulli_cdf`
- [x] `bernoulli_pmf`
- [x] `bernoulli_quantile`
- [x] `binomial_cdf`
- [x] `binomial_coef`
- [x] `binomial_pmf`
- [x] `binomial_quantile`
- [x] `discrete_uniform_cdf`
- [x] `discrete_uniform_pmf`
- [x] `discrete_uniform_quantile`
- [x] `geometric_cdf`
- [x] `geometric_pmf`
- [x] `geometric_quantile`
- [x] `hypergeom_cdf`
- [x] `hypergeom_pmf`
- [x] `hypergeom_quantile`
- [x] `log_binomial_coef`
- [x] `log_factorial`
- [x] `nbinom_cdf`
- [x] `nbinom_pmf`
- [x] `nbinom_quantile`
- [x] `poisson_cdf`
- [x] `poisson_pmf`
- [x] `poisson_quantile`

### special_functions.hpp (14)

- [x] `beta`
- [x] `betainc`
- [x] `betaincinv`
- [x] `erf`
- [x] `erfc`
- [x] `gammainc_lower`
- [x] `gammainc_lower_inv`
- [x] `gammainc_upper`
- [x] `lbeta`
- [x] `lgamma`
- [x] `norm_cdf`
- [x] `norm_quantile`
- [x] `norm_sf` — added as a public function when the upper-tail cancellation was fixed
- [x] `tgamma`

### basic_statistics.hpp (13)

- [x] `argmax`
- [x] `argmin`
- [x] `geometric_mean`
- [x] `harmonic_mean`
- [x] `logarithmic_mean`
- [x] `mean`
- [x] `median`
- [x] `mode`
- [x] `modes`
- [x] `sum`
- [x] `trimmed_mean`
- [x] `weighted_harmonic_mean`
- [x] `weighted_mean`

### dispersion_spread.hpp (15)

- [x] `coefficient_of_variation`
- [x] `geometric_stddev`
- [x] `iqr`
- [x] `mean_absolute_deviation`
- [x] `population_stddev`
- [x] `population_variance`
- [x] `range`
- [x] `sample_stddev`
- [x] `sample_variance`
- [x] `stddev`
- [x] `stdev`
- [x] `var`
- [x] `variance`
- [x] `weighted_stddev`
- [x] `weighted_variance`

### order_statistics.hpp (7)

- [x] `five_number_summary`
- [x] `maximum`
- [x] `minimum`
- [x] `percentile`
- [x] `quartiles`
- [x] `weighted_median`
- [x] `weighted_percentile`

### shape_of_distribution.hpp (6)

- [x] `kurtosis`
- [x] `population_kurtosis`
- [x] `population_skewness`
- [x] `sample_kurtosis`
- [x] `sample_skewness`
- [x] `skewness`

### correlation_covariance.hpp (7)

- [x] `covariance`
- [x] `kendall_tau`
- [x] `pearson_correlation`
- [x] `population_covariance`
- [x] `sample_covariance`
- [x] `spearman_correlation`
- [x] `weighted_covariance`

### distance_metrics.hpp (7)

- [x] `chebyshev_distance`
- [x] `cosine_distance`
- [x] `cosine_similarity`
- [x] `euclidean_distance`
- [x] `mahalanobis_distance` — two dimensions only (documented behaviour)
- [x] `manhattan_distance`
- [x] `minkowski_distance`

### linear_regression.hpp (11)

- [x] `adjusted_r_squared`
- [x] `compute_residual_diagnostics`
- [x] `compute_vif`
- [x] `confidence_interval_mean`
- [x] `correlation_matrix_determinant`
- [x] `multicollinearity_score`
- [x] `multiple_linear_regression`
- [x] `predict`
- [x] `prediction_interval_simple`
- [x] `r_squared`
- [x] `simple_linear_regression`

### glm.hpp (12)

- [x] `compute_glm_residuals`
- [x] `glm_fit`
- [x] `incidence_rate_ratios`
- [x] `logistic_regression`
- [x] `odds_ratios`
- [x] `odds_ratios_ci`
- [x] `overdispersion_test`
- [x] `poisson_regression`
- [x] `predict_count`
- [x] `predict_probability`
- [x] `pseudo_r_squared_mcfadden`
- [x] `pseudo_r_squared_nagelkerke`

### model_selection.hpp (14)

- [x] `aic`
- [x] `aic_linear`
- [x] `aicc`
- [x] `bic`
- [x] `bic_linear`
- [x] `create_cv_folds` — compared deterministically with shuffle = false
- [x] `cross_validate_linear` — compared deterministically with shuffle = false
- [x] `cv_lasso` — compared deterministically with shuffle = false
- [x] `cv_ridge` — compared deterministically with shuffle = false
- [x] `elastic_net_regression` — does not map onto glmnet the way the lasso does; compared against an independent implementation
- [x] `lasso_regression` — matches glmnet exactly
- [x] `loocv_linear`
- [x] `press_statistic`
- [x] `ridge_regression`

### effect_size.hpp (15)

- [x] `cohens_d`
- [x] `cohens_d_two_sample`
- [x] `cohens_h`
- [x] `d_to_r`
- [x] `eta_squared`
- [x] `glass_delta`
- [x] `hedges_correction_factor`
- [x] `hedges_g`
- [x] `hedges_g_two_sample`
- [x] `odds_ratio`
- [x] `omega_squared`
- [x] `partial_eta_squared`
- [x] `r_to_d`
- [x] `risk_ratio`
- [x] `t_to_r`

### estimation.hpp (15)

- [x] `ci_mean`
- [x] `ci_mean_diff`
- [x] `ci_mean_diff_pooled`
- [x] `ci_mean_diff_welch`
- [x] `ci_mean_z`
- [x] `ci_proportion`
- [x] `ci_proportion_diff`
- [x] `ci_proportion_wilson`
- [x] `ci_variance`
- [x] `margin_of_error_mean`
- [x] `margin_of_error_proportion`
- [x] `margin_of_error_proportion_worst_case`
- [x] `sample_size_for_moe_mean`
- [x] `sample_size_for_moe_proportion`
- [x] `standard_error`

### power_analysis.hpp (8)

- [x] `power_analysis_t_one_sample`
- [x] `power_analysis_t_one_sample_n`
- [x] `power_prop_test`
- [x] `power_t_test_one_sample`
- [x] `power_t_test_two_sample`
- [x] `sample_size_prop_test`
- [x] `sample_size_t_test_one_sample`
- [x] `sample_size_t_test_two_sample`

### survival.hpp (4)

- [x] `kaplan_meier`
- [x] `logrank_test`
- [x] `median_survival_time`
- [x] `nelson_aalen` — no tie correction, so it differs from R type = "fh"

### time_series.hpp (12)

- [x] `acf`
- [x] `autocorrelation`
- [x] `diff`
- [x] `exponential_moving_average`
- [x] `lag`
- [x] `mae`
- [x] `mape`
- [x] `moving_average`
- [x] `mse`
- [x] `pacf`
- [x] `rmse`
- [x] `seasonal_diff`

### robust.hpp (10)

- [x] `biweight_midvariance`
- [x] `cooks_distance`
- [x] `detect_outliers_iqr`
- [x] `detect_outliers_modified_zscore`
- [x] `detect_outliers_zscore`
- [x] `dffits`
- [x] `hodges_lehmann`
- [x] `mad`
- [x] `mad_scaled`
- [x] `winsorize`

### multivariate.hpp (7)

- [x] `correlation_matrix`
- [x] `covariance_matrix`
- [x] `min_max_scale`
- [x] `pca`
- [x] `pca_transform`
- [x] `power_iteration`
- [x] `standardize`

### categorical.hpp (5)

- [x] `contingency_table`
- [x] `number_needed_to_treat`
- [x] `odds_ratio`
- [x] `relative_risk`
- [x] `risk_difference`

### frequency_distribution.hpp (5)

- [x] `cumulative_frequency`
- [x] `cumulative_relative_frequency`
- [x] `frequency_count`
- [x] `frequency_table`
- [x] `relative_frequency`

### clustering.hpp (5)

- [x] `cut_dendrogram`
- [x] `euclidean_distance`
- [x] `hierarchical_clustering`
- [x] `manhattan_distance`
- [x] `silhouette_score`

### data_wrangling.hpp (27)

- [x] `argsort`
- [x] `bin_equal_freq`
- [x] `bin_equal_width`
- [x] `boxcox_transform`
- [x] `drop_duplicates`
- [x] `fillna_bfill`
- [x] `fillna_ffill`
- [x] `fillna_interpolate`
- [x] `fillna_mean`
- [x] `fillna_median`
- [x] `get_duplicates`
- [x] `group_count`
- [x] `group_mean`
- [x] `group_sum`
- [x] `label_encode`
- [x] `log1p_transform`
- [x] `log_transform`
- [x] `one_hot_encode`
- [x] `rank_transform`
- [x] `rolling_max`
- [x] `rolling_mean`
- [x] `rolling_min`
- [x] `rolling_std`
- [x] `rolling_sum`
- [x] `sort_values`
- [x] `sqrt_transform`
- [x] `value_counts`

### missing_data.hpp (6)

- [x] `analyze_missing_patterns` — cross-checked with mice::md.pattern
- [x] `correlation_matrix_pairwise`
- [x] `create_missing_indicator`
- [x] `extract_complete_cases`
- [x] `impute_conditional_mean` — uses only the first predictor (documented behaviour)
- [x] `test_mcar_simple` — not Little's test, and not what naniar reports

### numerical_utils.hpp (3)

- [x] `expm1_safe`
- [x] `kahan_sum`
- [x] `log1p_safe`
