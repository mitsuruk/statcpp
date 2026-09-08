# R Comparability Inventory

Every public function in `include/statcpp/*.hpp` classified by whether its result
can be compared numerically against R 4.4.2.

## Totals

The set is the public functions directly inside `namespace statcpp`, as reported
by Doxygen XML. The 24 functions in `statcpp::detail` are implementation details
and are out of scope.

|Measure|Value|
|---|---|
|Public function declarations (overloads included)|538|
|Public function unique names|386|
|Public function entries by module|391|
|`statcpp::detail` (out of scope)|24|

The module-entry count is 391 rather than 386 because five names, such as
`euclidean_distance`, appear in two headers. This inventory classifies by module.

|Category|Functions|Share|
|---|---|---|
|Comparable|321|82.1%|
|Not comparable|70|17.9%|
|Total|391|100%|

## Licensing

statcpp is MIT licensed; R and the R packages used here are under GPL-family
licenses. **The following rules keep the two apart.**

- R runs as a **separate process** (`Rscript`). No library is linked.
- Only **numeric data** crosses the boundary. The output of a GPL program is not
  itself covered by the GPL.
- **R source code and algorithm implementations are never ported into statcpp.**
  That is the only real contamination path.
- Linking `libRmath` was rejected, because its GPL-2 would propagate into the
  test binary.
- The generated reference files are factual data and stay distributable as MIT.

## Environment

macOS only. R 4.4.2 on aarch64-apple-darwin20.

---

## 1. Comparable functions

### 1.1 parametric_tests.hpp (14/14)

|statcpp|R counterpart|
|---|---|
|`t_test`|`t.test(x, mu=)`|
|`t_test_two_sample`|`t.test(x, y, var.equal=TRUE)`|
|`t_test_welch`|`t.test(x, y)`|
|`t_test_paired`|`t.test(x, y, paired=TRUE)`|
|`z_test`|closed form (`pnorm`)|
|`f_test`|`var.test(x, y)`|
|`chisq_test_gof`|`chisq.test(obs, p=)`|
|`chisq_test_gof_uniform`|`chisq.test(obs)`|
|`chisq_test_independence`|`chisq.test(m, correct=FALSE)`|
|`z_test_proportion`|`prop.test(x, n, p, correct=FALSE)`|
|`z_test_proportion_two_sample`|`prop.test(c(x1,x2), c(n1,n2))`|
|`bonferroni_correction`|`p.adjust(method="bonferroni")`|
|`holm_correction`|`p.adjust(method="holm")`|
|`benjamini_hochberg_correction`|`p.adjust(method="BH")`|

### 1.2 nonparametric_tests.hpp (10/11)

|statcpp|R counterpart|
|---|---|
|`wilcoxon_signed_rank_test`|`wilcox.test(x, mu=)`|
|`mann_whitney_u_test`|`wilcox.test(x, y)`|
|`kruskal_wallis_test`|`kruskal.test`|
|`shapiro_wilk_test`|`shapiro.test`|
|`ks_test_normal`|`nortest::lillie.test` (deprecated alias of `lilliefors_test`)|
|`lilliefors_test`|`nortest::lillie.test`|
|`bartlett_test`|`bartlett.test`|
|`levene_test`|`car::leveneTest`|
|`fisher_exact_test`|`fisher.test`|
|`compute_ranks_with_ties`|`rank(ties.method="average")`|

### 1.3 anova.hpp (13/13)

|statcpp|R counterpart|
|---|---|
|`one_way_anova`|`summary(aov(y ~ g))`|
|`two_way_anova`|`summary(aov(y ~ a*b))`|
|`one_way_ancova`|`car::Anova(lm(y ~ x + g), type = 2)`|
|`tukey_hsd`|`TukeyHSD(aov(...))`|
|`dunnett_posthoc`|closed form (Bonferroni-adjusted t, not classical Dunnett)|
|`bonferroni_posthoc`|`pairwise.t.test(p.adjust="bonferroni")`|
|`scheffe_posthoc`|closed form (critical value from `qf`)|
|`eta_squared`|`effectsize::eta_squared`|
|`omega_squared`|`effectsize::omega_squared`|
|`partial_eta_squared_a`|`effectsize::eta_squared(partial=TRUE)`|
|`partial_eta_squared_b`|as above|
|`partial_eta_squared_interaction`|as above|
|`cohens_f`|`effectsize::cohens_f`|

### 1.4 continuous_distributions.hpp (32/42)

Ten distributions times pdf/cdf/quantile gives 30, plus `studentized_range_cdf`
to `ptukey` and `studentized_range_quantile` to `qtukey`.

|statcpp suffix|R prefix|Distributions|
|---|---|---|
|`_pdf`|`d`|normal, t, chisq, f, gamma, beta, exponential, uniform, lognormal, weibull|
|`_cdf`|`p`|as above|
|`_quantile`|`q`|as above|

### 1.5 discrete_distributions.hpp (24/31)

|statcpp suffix|R prefix|Distributions|
|---|---|---|
|`_pmf`|`d`|binomial, poisson, geometric, hypergeom, nbinom, bernoulli, discrete_uniform|
|`_cdf`|`p`|as above|
|`_quantile`|`q`|as above|

`bernoulli_*` is compared with `dbinom(size=1)` and `discrete_uniform_*` with a
closed form. The helpers map as `binomial_coef` to `choose`, `log_binomial_coef`
to `lchoose` and `log_factorial` to `lfactorial`.

### 1.6 basic_statistics.hpp (13/14)

|statcpp|R counterpart|
|---|---|
|`sum`|`sum`|
|`mean`|`mean`|
|`median`|`median`|
|`mode` / `modes`|`names(sort(table(x), decreasing=TRUE))`|
|`geometric_mean`|`exp(mean(log(x)))`|
|`harmonic_mean`|`1/mean(1/x)`|
|`weighted_mean`|`weighted.mean`|
|`weighted_harmonic_mean`|`sum(w)/sum(w/x)`|
|`logarithmic_mean`|`(b-a)/(log(b)-log(a))`|
|`trimmed_mean`|`mean(x, trim=)` **check the trimming rule**|
|`argmax` / `argmin`|`which.max` / `which.min` (zero- versus one-based)|

### 1.7 dispersion_spread.hpp (15/15)

|statcpp|R counterpart|
|---|---|
|`var` / `stdev`|`var` / `sd` **ddof defaults to 0, so the bare call is the population form**|
|`variance` / `sample_variance`|`var`|
|`population_variance`|`var(x)*(n-1)/n`|
|`stddev` / `sample_stddev`|`sd`|
|`population_stddev`|`sd(x)*sqrt((n-1)/n)`|
|`weighted_variance` / `weighted_stddev`|`cov.wt(method="unbiased")`|
|`range`|`diff(range(x))`|
|`iqr`|`IQR(x, type=7)`|
|`mean_absolute_deviation`|`mean(abs(x-mean(x)))` **not R's `mad()`**|
|`coefficient_of_variation`|`sd(x)/abs(mean(x))`|
|`geometric_stddev`|`exp(sd(log(x)))`|

### 1.8 order_statistics.hpp (7/8)

|statcpp|R counterpart|
|---|---|
|`minimum` / `maximum`|`min` / `max`|
|`percentile`|`quantile(type=7)`|
|`quartiles`|`quantile(probs=c(.25,.5,.75), type=7)`|
|`five_number_summary`|`quantile(type=7)` **not `fivenum`, which reports hinges**|
|`weighted_median` / `weighted_percentile`|closed form; equals `quantile(type=2)` at unit weights|

### 1.9 shape_of_distribution.hpp (6/6)

|statcpp|R counterpart|
|---|---|
|`skewness` / `sample_skewness`|`e1071::skewness(type=2)`|
|`population_skewness`|`e1071::skewness(type=1)`|
|`kurtosis` / `sample_kurtosis`|`e1071::kurtosis(type=2)`|
|`population_kurtosis`|`e1071::kurtosis(type=1)`|

### 1.10 correlation_covariance.hpp (7/7)

|statcpp|R counterpart|
|---|---|
|`pearson_correlation`|`cor(method="pearson")`|
|`spearman_correlation`|`cor(method="spearman")`|
|`kendall_tau`|`cor(method="kendall")` (tau-b)|
|`covariance` / `sample_covariance`|`cov`|
|`population_covariance`|`cov(x,y)*(n-1)/n`|
|`weighted_covariance`|`cov.wt(method="unbiased")`|

### 1.11 distance_metrics.hpp (7/7)

The superseded `NON_VERIFIABLE_FUNCTIONS.md` classified these as having no R
counterpart. That was **wrong**: `dist()` covers all of them.

|statcpp|R counterpart|
|---|---|
|`euclidean_distance`|`dist(method="euclidean")`|
|`manhattan_distance`|`dist(method="manhattan")`|
|`chebyshev_distance`|`dist(method="maximum")`|
|`minkowski_distance`|`dist(method="minkowski", p=)`|
|`mahalanobis_distance`|`sqrt(mahalanobis(...))` **two dimensions only**|
|`cosine_similarity` / `cosine_distance`|`proxy::simil(method="cosine")`|

### 1.12 linear_regression.hpp (11/11)

|statcpp|R counterpart|
|---|---|
|`simple_linear_regression` / `multiple_linear_regression`|`lm`|
|`predict`|`predict(lm)`|
|`r_squared` / `adjusted_r_squared`|`summary(lm)$r.squared` / `$adj.r.squared`|
|`confidence_interval_mean`|`predict(interval="confidence")`|
|`prediction_interval_simple`|`predict(interval="prediction")`|
|`compute_vif` / `multicollinearity_score`|`car::vif` / `1 - abs(det(cor(X)))`|
|`compute_residual_diagnostics`|`residuals`/`sigma`, `rstandard`, `hatvalues`, `cooks.distance`|
|`correlation_matrix_determinant`|`det(cor(X))`|

### 1.13 glm.hpp (12/12)

|statcpp|R counterpart|
|---|---|
|`glm_fit` / `logistic_regression`|`glm(family=binomial)`|
|`poisson_regression`|`glm(family=poisson)`|
|`predict_probability` / `predict_count`|`predict(type="response")`|
|`odds_ratios` / `incidence_rate_ratios`|`exp(coef(m))[-1]` **intercept excluded**|
|`odds_ratios_ci`|`exp(confint.default(m))[-1, ]`|
|`pseudo_r_squared_mcfadden`|`1 - logLik(m)/logLik(m0)`|
|`pseudo_r_squared_nagelkerke`|`performance::r2_nagelkerke`|
|`overdispersion_test`|Pearson chi-square divided by residual df|
|`compute_glm_residuals`|`residuals(type = "deviance"/"pearson"/"response")`|

### 1.14 effect_size.hpp (15/18)

|statcpp|R counterpart|
|---|---|
|`cohens_d` / `cohens_d_two_sample`|`effectsize::cohens_d`|
|`hedges_g` / `hedges_g_two_sample`|closed form **`effectsize` uses the exact gamma factor**|
|`hedges_correction_factor`|`1 - 3/(4*df-1)`|
|`glass_delta`|`effectsize::glass_delta`|
|`cohens_h`|`2*asin(sqrt(p1)) - 2*asin(sqrt(p2))`|
|`eta_squared` / `partial_eta_squared` / `omega_squared`|`effectsize::*`|
|`odds_ratio` / `risk_ratio`|closed form|
|`d_to_r` / `r_to_d` / `t_to_r`|`effectsize::d_to_r` and friends|

### 1.15 estimation.hpp (15/15)

|statcpp|R counterpart|
|---|---|
|`ci_mean`|`t.test(x)$conf.int`|
|`ci_mean_z`|closed form (`qnorm`)|
|`ci_mean_diff` / `ci_mean_diff_pooled`|`t.test(x, y, var.equal=TRUE)$conf.int` **`ci_mean_diff` is the pooled form**|
|`ci_mean_diff_welch`|`t.test(x, y)$conf.int`|
|`ci_proportion`|Wald closed form|
|`ci_proportion_wilson`|`prop.test(correct=FALSE)$conf.int`|
|`ci_proportion_diff`|closed form|
|`ci_variance`|`(n-1)*var(x)/qchisq(...)`|
|`standard_error`|`sd(x)/sqrt(n)`|
|`margin_of_error_*` / `sample_size_for_moe_*`|closed form|

### 1.16 power_analysis.hpp (8/8)

|statcpp|R counterpart|
|---|---|
|`power_t_test_one_sample` / `power_t_test_two_sample`|closed form **normal approximation, not `power.t.test`**|
|`sample_size_t_test_one_sample` / `_two_sample`|closed form plus incremental search|
|`power_analysis_t_one_sample` / `_n`|as above|
|`power_prop_test` / `sample_size_prop_test`|`power.prop.test` (same closed form)|

### 1.17 special_functions.hpp (14/16)

|statcpp|R counterpart|
|---|---|
|`erf` / `erfc`|`2*pnorm(x*sqrt(2))-1` / `2*pnorm(-x*sqrt(2))`|
|`lgamma` / `tgamma`|`lgamma` / `gamma`|
|`beta` / `lbeta`|`beta` / `lbeta`|
|`betainc` / `betaincinv`|`pbeta` / `qbeta`|
|`gammainc_lower` / `gammainc_upper`|`pgamma(lower.tail=TRUE/FALSE)`|
|`gammainc_lower_inv`|`qgamma`|
|`norm_cdf` / `norm_quantile`|`pnorm` / `qnorm`|
|`norm_sf`|`pnorm(lower.tail=FALSE)`|

### 1.18 survival.hpp (4/4)

|statcpp|R counterpart|
|---|---|
|`kaplan_meier`|`survival::survfit(Surv(t,e) ~ 1, conf.type="plain")`|
|`nelson_aalen`|`cumsum(d/n)` **no tie correction, unlike `type="fh"`**|
|`median_survival_time`|`summary(survfit)$table["median"]`|
|`logrank_test`|`survival::survdiff`|

### 1.19 time_series.hpp (12/12)

|statcpp|R counterpart|
|---|---|
|`acf` / `autocorrelation`|`acf(plot=FALSE)`|
|`pacf`|`pacf(plot=FALSE)` **statcpp also reports lag 0 as 1**|
|`diff` / `seasonal_diff`|`diff(lag=)`|
|`lag`|closed form|
|`moving_average`|`stats::filter(sides=1)`|
|`exponential_moving_average`|closed form (recursion)|
|`mse` / `rmse` / `mae` / `mape`|closed form|

### 1.20 robust.hpp (10/10)

|statcpp|R counterpart|
|---|---|
|`mad`|`mad(x, constant=1)`|
|`mad_scaled`|`mad(x)` (constant 1.4826)|
|`winsorize`|closed form (clipped at `quantile`)|
|`hodges_lehmann`|`wilcox.test(conf.int=TRUE)$estimate`|
|`biweight_midvariance`|closed form (Tukey biweight, squared denominator)|
|`detect_outliers_zscore` / `_modified_zscore` / `_iqr`|closed form|
|`cooks_distance`|`cooks.distance(lm)`|
|`dffits`|`dffits(lm)`|

### 1.21 multivariate.hpp (7/7)

|statcpp|R counterpart|
|---|---|
|`correlation_matrix` / `covariance_matrix`|`cor` / `cov`|
|`standardize`|`scale`|
|`min_max_scale`|closed form|
|`pca` / `pca_transform`|`prcomp` **eigenvector signs are arbitrary**|
|`power_iteration`|`eigen()$values[1]` (leading eigenvalue only)|

### 1.22 categorical.hpp (5/5)

|statcpp|R counterpart|
|---|---|
|`contingency_table`|`table`|
|`odds_ratio`|`(a*d)/(b*c)` **not the conditional MLE `fisher.test` reports**|
|`relative_risk` / `risk_difference` / `number_needed_to_treat`|closed form|

Interval bounds in this header use a hard-coded z = 1.96.

### 1.23 frequency_distribution.hpp (5/5)

|statcpp|R counterpart|
|---|---|
|`frequency_count` / `frequency_table`|`table`|
|`relative_frequency`|`prop.table(table(x))`|
|`cumulative_frequency`|`cumsum(table(x))`|
|`cumulative_relative_frequency`|`cumsum(prop.table(table(x)))`|

### 1.24 clustering.hpp (5/7)

|statcpp|R counterpart|
|---|---|
|`hierarchical_clustering`|`hclust` (deterministic)|
|`cut_dendrogram`|`cutree`|
|`silhouette_score`|`cluster::silhouette` (given a fixed assignment)|
|`euclidean_distance` / `manhattan_distance`|`dist`|

### 1.25 model_selection.hpp (14/15)

|statcpp|R counterpart|
|---|---|
|`aic` / `aic_linear`|`AIC(lm)`|
|`bic` / `bic_linear`|`BIC(lm)`|
|`aicc`|`AIC + 2k(k+1)/(n-k-1)`|
|`ridge_regression`|closed form on population-standardised predictors|
|`lasso_regression`|`glmnet(alpha=1, lambda = lambda/n)` **exact match**|
|`elastic_net_regression`|independent coordinate descent on statcpp's own objective|
|`loocv_linear`|algebraic leave-one-out shortcut|
|`press_statistic`|`sum((resid(m)/(1-hatvalues(m)))^2)`|
|`create_cv_folds` / `cross_validate_linear` / `cv_ridge` / `cv_lasso`|deterministic with `shuffle = false`|

### 1.26 data_wrangling.hpp (27/39)

|statcpp|R counterpart|
|---|---|
|`rank_transform`|`rank`|
|`rolling_mean` / `_sum` / `_min` / `_max` / `_std`|one value per complete window|
|`log_transform` / `log1p_transform` / `sqrt_transform`|`log` / `log1p` / `sqrt`|
|`boxcox_transform`|`(x^l - 1)/l`|
|`bin_equal_width` / `bin_equal_freq`|zero-based bin index per observation|
|`group_mean` / `group_sum` / `group_count`|`tapply`|
|`value_counts`|`table`|
|`one_hot_encode` / `label_encode`|first-appearance class order, **not sorted levels**|
|`fillna_mean` / `_median`|closed form|
|`fillna_ffill` / `_bfill`|`zoo::na.locf`|
|`fillna_interpolate`|`zoo::na.approx`|
|`sort_values` / `argsort`|`sort` / `order` (zero-based)|
|`drop_duplicates` / `get_duplicates`|`unique` / `duplicated` (order unspecified)|

### 1.27 missing_data.hpp (6/12)

|statcpp|R counterpart|
|---|---|
|`extract_complete_cases`|`complete.cases`|
|`create_missing_indicator`|`is.na`|
|`correlation_matrix_pairwise`|`cor(use="pairwise.complete.obs")`|
|`impute_conditional_mean`|simple `lm` on the **first predictor only**|
|`analyze_missing_patterns`|`colMeans(is.na)` and `table`, cross-checked with `mice::md.pattern`|
|`test_mcar_simple`|reproduction of statcpp's own heuristic, **not Little's test**|

### 1.28 numerical_utils.hpp (3/15)

|statcpp|R counterpart|
|---|---|
|`log1p_safe`|`log1p`|
|`expm1_safe`|`expm1`|
|`kahan_sum`|`sum` (the point is the accuracy difference)|

---

## 2. Functions brought into scope by additional work

Six functions were originally classified as conditional, requiring R packages
that were not installed, and four more depended on randomly shuffled folds. All
of them are now compared.

|statcpp|What made it comparable|
|---|---|
|`lasso_regression`|`glmnet` installed; matches exactly with `s = lambda / n`|
|`elastic_net_regression`|`glmnet` installed, but the mapping does not hold; compared against an independent implementation of statcpp's objective|
|`analyze_missing_patterns`|`mice` installed for the cross-check|
|`test_mcar_simple`|`naniar` installed for contrast; the reference reproduces statcpp's own procedure|
|`create_cv_folds` / `cross_validate_linear` / `cv_ridge` / `cv_lasso`|a `shuffle` argument was added, so `false` gives contiguous deterministic folds|

This moved the comparable count from 313 to 321.

---

## 3. Functions that are not comparable (70)

### 3.1 Random, and therefore non-deterministic (34)

R and C++ draw from different random number generator streams, so fixing a seed
does not align the sequences. Comparing values is impossible in principle.

|Module|Functions|
|---|---|
|`random_engine.hpp`|`get_random_engine`, `set_seed`, `randomize_seed`|
|`continuous_distributions.hpp`|`normal_rand`, `t_rand`, `chisq_rand`, `f_rand`, `gamma_rand`, `beta_rand`, `exponential_rand`, `uniform_rand`, `lognormal_rand`, `weibull_rand`|
|`discrete_distributions.hpp`|`binomial_rand`, `poisson_rand`, `geometric_rand`, `hypergeom_rand`, `nbinom_rand`, `bernoulli_rand`, `discrete_uniform_rand`|
|`resampling.hpp`|`bootstrap`, `bootstrap_sample`, `bootstrap_mean`, `bootstrap_median`, `bootstrap_stddev`, `bootstrap_bca`, `permutation_test_two_sample`, `permutation_test_paired`, `permutation_test_correlation`|
|`clustering.hpp`|`kmeans`, `kmeans_plusplus_init`|
|`data_wrangling.hpp`|`sample_with_replacement`, `sample_without_replacement`, `stratified_sample`|

**Alternative**: check the statistical properties instead, such as convergence of
the mean and variance at large samples and a Kolmogorov-Smirnov goodness of fit.
That compares against theory rather than against R.

### 3.2 Numerical utilities, not statistics (12)

Helpers for floating-point comparison and convergence testing. R has no
corresponding concept.

|Module|Functions|
|---|---|
|`numerical_utils.hpp`|`approx_equal`, `approx_equal_range`, `is_finite`, `all_finite`, `is_zero`, `clamp`, `in_range`, `safe_divide`, `relative_error`, `has_converged`, `has_converged_abs`, `has_converged_rel`|

**Alternative**: boundary and error cases are already covered by the unit tests
under `test/`.

### 3.3 Container operations and predicates, not statistics (11)

Data structure manipulation, so there is nothing numeric to compare.

|Module|Functions|
|---|---|
|`basic_statistics.hpp`|`count`|
|`data_wrangling.hpp`|`filter`, `filter_range`, `filter_rows`, `is_na`, `dropna`, `fillna`, `validate_data`, `validate_range`, `group_by`|
|`model_selection.hpp`|`generate_lambda_grid`|

### 3.4 Implementation helpers, covered through the public API (4)

These live in the `statcpp` namespace but are implementation bodies. They have no
R counterpart on their own, and comparing the caller covers them.

|Module|Functions|Reason|
|---|---|---|
|`special_functions.hpp`|`lgamma_impl`, `betainc_impl`|bodies of `lgamma` and `betainc`|
|`nonparametric_tests.hpp`|`compute_tie_groups`|internal step of the rank tests|
|`order_statistics.hpp`|`interpolate_at`|internal interpolation of `percentile`|

### 3.5 Interpretation functions returning strings (3)

They report the magnitude of an effect size as text. The thresholds are Cohen's
conventions, an arbitrary convention with no normative implementation in R.

|Module|Functions|
|---|---|
|`effect_size.hpp`|`interpret_cohens_d`, `interpret_correlation`, `interpret_eta_squared`|

**Alternative**: the branch at each threshold is covered by the unit tests under
`test/`.

### 3.6 statcpp-specific procedures with no R counterpart (6)

The missing-data sensitivity analyses are formulated differently in every
implementation, and R has no standard equivalent. All are in `missing_data.hpp`.

|Function|Reason|
|---|---|
|`diagnose_missing_mechanism`|the MCAR/MAR/MNAR decision logic is specific to statcpp|
|`sensitivity_analysis_pattern_mixture`|the pattern-mixture formulation is specific to statcpp|
|`sensitivity_analysis_selection_model`|the selection-model formulation is specific to statcpp|
|`find_tipping_point`|the threshold search is specific to statcpp|
|`multiple_imputation_bootstrap`|random, and specific to statcpp|
|`multiple_imputation_pmm`|as above|

---

## 3.7 Out of scope: the statcpp::detail namespace (24)

Non-public implementation details. Comparing the public callers covers them
indirectly.

|Module|Functions|
|---|---|
|`linear_regression.hpp`|`cholesky`, `inverse_cholesky`, `solve_cholesky`, `matrix_multiply`, `matrix_vector_multiply`, `transpose`, `validate_matrix_structure`, `validate_no_intercept_column`|
|`glm.hpp`|`link_transform`, `inverse_link`, `link_derivative`, `variance_function`, `deviance_residual`, `solve_weighted_least_squares`|
|`power_analysis.hpp`|`critical_z_one_sided`, `critical_z_two_sided`, `critical_t_one_sided`, `critical_t_two_sided`, `noncentrality_parameter_t`, `noncentrality_parameter_t_two_sample`, `alternative_to_string`|
|`model_selection.hpp`|`standardize_features`, `rescale_coefficients`|
|`correlation_covariance.hpp`|`compute_ranks`|

---

## 4. Definition differences that matter more than rounding

These outrank numerical error. When the **definitions themselves differ**, no
amount of tightening the tolerance will make the two agree.

|Item|R default|What to watch|
|---|---|---|
|Quantiles|`quantile(type=7)`|There are nine types. State which one statcpp follows.|
|`fivenum`|hinges|A different definition from `quantile(type=7)`. Do not conflate them.|
|Mean absolute deviation|—|R's `mad()` is the **median** absolute deviation, a different statistic from `mean_absolute_deviation`.|
|`var` / `stdev`|sample form|statcpp's ddof defaults to 0, so the bare call is the population form.|
|`trimmed_mean`|`mean(trim=)`|R drops `floor(n*trim)` from each end. Confirm the rounding rule matches.|
|Skewness and kurtosis|`e1071` `type`|Types 1, 2 and 3 give different values, and kurtosis may or may not be excess.|
|Wilcoxon family|`correct=TRUE`|Continuity correction is on by default.|
|`chisq.test`|`correct=TRUE` (2x2)|Yates' correction is on by default for 2x2 tables.|
|Odds ratio|`fisher.test`|Reports a conditional maximum likelihood estimate, not `ad/bc`.|
|`AIC`|normal likelihood|For a linear model the variance counts as one parameter.|
|PCA|`prcomp`|Eigenvector signs are arbitrary; normalise before comparing.|
|`range`|`c(min,max)`|statcpp returns the width, so compare with `diff(range(x))`.|
|`argmax`|`which.max`|R is one-based, C++ is zero-based.|
|`mahalanobis`|squared distance|statcpp returns the distance itself, and supports two dimensions only.|
|`pacf`|starts at lag 1|statcpp reports PACF(0) = 1 as well.|
|`label_encode`|—|Codes follow first appearance, not sorted factor levels.|

---

## 5. What this inventory replaced

|Item|Superseded documents|This inventory|
|---|---|---|
|Coverage|57 functions|391 entries, every public function|
|`distance_metrics.hpp`|"no R counterpart", not comparable|all seven are comparable through `dist()`|
|`clustering.hpp`|treated as not comparable|`hclust`, `cutree` and `silhouette` are deterministic and comparable|
|`special_functions.hpp`|mostly treated as not comparable|13 are comparable through `pbeta`, `pgamma` and friends|

`VERIFIED_FUNCTIONS.md` and `NON_VERIFIABLE_FUNCTIONS.md` were removed in favour
of this file and `VERIFICATION_CHECKLIST.md`.

---

## 6. Corrections to the published figures

Building this inventory required measuring the library, which showed that the
figures in `README.md`, `README.ja.md` and both `API_REFERENCE.md` files were
wrong. They have since been corrected. Measurement used Doxygen XML
(`memberdef kind="function"` under `namespace statcpp`) and an actual test run.

|Item|Previously published|Measured|
|---|---|---|
|Public functions|524|538 declarations, 386 unique names|
|Unit tests|793|857|
|Verification tests|167 checks over 57 functions|164 tests over 321 functions|
|Header files|31|31|

### Where 524 came from

`524` entered the README in commit `4871469` (Version 0.2.0). Extracting the
headers at that commit and measuring them with Doxygen gives 537 declarations and
385 unique names, so the figure did not match even when it was written. Between
v0.2.0 and the following release six files under `include/statcpp` changed, but
the diff added and removed **no functions**.

No alternative counting rule produces 524 either.

|Counting rule|Value|
|---|---|
|`statcpp` public functions (declarations, overloads included)|537|
|`statcpp` public functions (unique names)|385|
|`statcpp` plus `statcpp::detail`|562|
|Rows in the `docs/API_REFERENCE.md` tables|318|
|Unique function names in that file|313|
|The `include-ja` variant|537 / 385, identical to `include`|

The basis for 524 could not be established.

### API reference coverage

At the time of the audit `API_REFERENCE.md` listed 313 unique function names
against 385 public functions. The gap was almost entirely the distribution
families, which are documented in wide tables that a first-column scan misses.
The genuinely missing entries were `norm_sf`, `betainc_impl` and `lgamma_impl`,
which have since been added.
