# Cross-Verification with R — Methodology

## Scope

Every one of the 321 R-comparable public functions is compared against R 4.4.2.
The remaining public functions are not comparable for the reasons recorded in
`R_VERIFICATION_INVENTORY.md`: they are non-deterministic, they are container
utilities rather than statistics, or R has no counterpart.

The earlier standalone `verify_vs_r` program, which hard-coded 167 reference
values for 57 functions, has been removed: this system covers everything it did
and generates its reference values instead of transcribing them.

## GTest system

### Pipeline

```text
generate_r_reference.R          R owns both the input data and the expected results
        |  Rscript (development only, macOS)
        v
r_reference_*.hpp               generated, committed, hexadecimal float literals
        |
        v
test_vs_r_*.cpp (Google Test)   R is not needed to build or run
```

R is the single source of truth for the input data as well as the expected
values, so there is no manual transcription step and the two sides cannot drift
apart. The generated headers are committed, so neither building nor running the
tests requires R.

### Licensing

statcpp is MIT licensed; R and the R packages used here are under GPL-family
licenses. R is never linked into the test binary — it runs as a separate process
at generation time and only numbers cross the boundary, which does not create a
derivative work. The one thing that would contaminate the license is porting R's
source or algorithms into statcpp, which is not done.

### Exact value transfer

Reference values are written with R's `sprintf("%a", x)` and read back as C++17
hexadecimal floating-point literals, so the constants are bit-identical to the
values R computed:

```text
sprintf("%a",    qnorm(0.975))  ->  0x1.f5c0331eeff82p+0   exact
sprintf("%.15g", qnorm(0.975))  ->  1.95996398454005       lossy
```

Any difference a test reports is therefore a genuine algorithmic difference
between statcpp and R, not an artefact of writing the number down.

### Tolerances

Comparison uses a combined bound, `|actual - expected| <= atol + rtol * |expected|`.
A pure relative bound is not usable: `pnorm(37.10, log.p = TRUE)` is about
-1.4e-301, so an absolute difference of 3e-308 shows up as a relative error of
2.3e-07.

Each generated case carries two relative tolerances:

|Constant|Default|Applies to|
|---|---|---|
|`kRtol`|1e-12|plain arithmetic: statistics, sums of squares, degrees of freedom|
|`kPRtol`|1e-8|values passing through statcpp's own distribution functions|

The p-value tier is looser because statcpp implements its own incomplete beta
and incomplete gamma functions. 1e-8 is calibrated from the largest deviation
actually observed in this suite (2.1e-09, the ANCOVA covariate p-value).

Cases that need a wider bound record the reason in the generated struct's
Doxygen comment rather than at the call site.

### Degenerate reference values

A reference value of exactly zero cannot be matched under a relative bound, so
input data must be chosen to avoid it. Two cases in this suite were rebuilt for
that reason: the ANOVA groups have identical spreads, which makes both the
Bartlett and Levene statistics exactly zero, and an additive two-way layout
makes the interaction sum of squares exactly zero.

### Regenerating

```bash
Rscript generate_r_reference.R      # or: cmake --build <dir> --target regenerate_r_reference
```

Requires R 4.4.2 with `nortest`, `car`, `effectsize`, `e1071`, `proxy`, `performance`,
`survival`, `cluster`, `zoo`, `glmnet`, `mice` and `naniar`. The output is
deterministic: re-running must not change the generated headers.

### Definition differences found so far

|Function|Difference|
|---|---|
|`lilliefors_test`, `ks_test_normal`|Resolved. statcpp now uses the Dallal and Wilkinson (1986) approximation and reproduces `nortest::lillie.test` exactly for p <= 0.10. Above that level the approximation is out of its published range, so that case asserts only that the p-value stays above every conventional significance level.|
|`one_way_ancova`|statcpp reports partial (type II) sums of squares, so the reference is `car::Anova(type = 2)`, not R's sequential `summary(aov(y ~ x + g))`.|
|`scheffe_posthoc`|`posthoc_comparison::statistic` holds the signed t statistic; `F = t^2 / df_between` is used only for the p-value.|
|`dunnett_posthoc`|Applies a Bonferroni approximation rather than the classical Dunnett procedure, so it does not match `multcomp::glht(..., mcp(g = "Dunnett"))`.|
|`tukey_hsd`|`TukeyHSD` reports `mean[j] - mean[i]`; statcpp reports `mean[i] - mean[j]`, so signs are flipped and interval bounds swapped.|
|`ks_test_normal`|Deprecated in statcpp; it forwards to `lilliefors_test`.|
|`var`, `stdev`|Their ddof argument defaults to 0, so the bare call is the population form. R's `var`/`sd` are the sample form.|
|`mean_absolute_deviation`|Mean absolute deviation about the mean. R's `mad()` is the median absolute deviation, a different statistic.|
|`five_number_summary`|Uses type 7 quantiles, not R's `fivenum()`, which reports Tukey hinges.|
|`weighted_variance`, `weighted_covariance`|Reliability-weight estimators, matching `cov.wt(method = "unbiased")`. They differ from `Hmisc::wtd.var`, which treats weights as frequencies.|
|`weighted_percentile`|A step function with averaging at exact cumulative-weight hits. With unit weights it coincides with `quantile(type = 2)`.|
|`mahalanobis_distance`|Supports two dimensions only (documented) and returns the distance, whereas R's `mahalanobis()` returns the squared distance.|
|`mode`|Resolves ties to the smallest value, since it iterates an ordered map.|
|p-values from z statistics|Resolved. Previously formed as `2 (1 - norm_cdf(abs(z)))`, which cancelled in the upper tail and returned exactly zero once abs(z) reached 8.30. statcpp now provides `norm_sf`, the survival function evaluated through erfc, and uses it at every upper-tail site. It agrees with R to 1.9e-13 relative out to x = 37.5. The Poisson intercept p-value went from 0 to 1.2688719e-117 against R 1.2688719e-117.|
|`power_t_test_*`|Documented normal approximation, not the noncentral t distribution R's `power.t.test` uses. Here 0.7819 against R's 0.7540, and a required n of 32 against 34.|
|`ci_mean_diff`|Delegates to the pooled, equal-variance interval. `ci_mean_diff_welch` is the separate-variance form.|
|`odds_ratios`, `odds_ratios_ci`, `incidence_rate_ratios`|Exclude the intercept.|
|`hedges_g`|Uses the `1 - 3/(4 df - 1)` approximation; the effectsize package uses the exact gamma-based factor, differing by about 1e-04.|
|`standardized_residuals`|Divides by sigma alone. `studentized_residuals` is R's `rstandard`; statcpp has no externally studentized residual.|
|`cross_validate_linear`, `cv_ridge`, `cv_lasso`|Shuffle folds with the global RNG by default. A `shuffle` argument was added so that passing `false` produces contiguous, reproducible folds; the tests use that.|
|`lasso_regression`|Matches `glmnet` exactly once the penalty is mapped as `s = lambda / n`: glmnet scales the residual sum of squares by 1/(2n) and statcpp by 1/2.|
|`elastic_net_regression`|Does NOT map onto glmnet the same way; its ridge term is applied on a different scale. Compared against an independent solution of the objective statcpp defines.|
|`test_mcar_simple`|Not Little's MCAR test. It accumulates a Welch statistic over ordered column pairs and converts the total with a Wilson-Hilferty approximation, so it does not match `naniar::mcar_test`.|
|`impute_conditional_mean`|Takes a vector of predictor columns but uses only its first element, fitting a simple linear regression. The complete-case filter still requires every listed predictor to be present.|
|`nelson_aalen`|Accumulates d/n directly. R's `type = "fh"` applies the Fleming-Harrington tie correction, so the two differ wherever events are tied.|
|`kaplan_meier`|Prepends the origin, so its first row is t = 0 with S = 1.|
|`label_encode`, `one_hot_encode`|Assign codes and columns in order of first appearance, not by sorted level.|
|`get_duplicates`|Returns its values in unspecified order.|
|`pacf`|Reports PACF(0) = 1; R's `pacf` starts at lag 1.|
|`odds_ratio`, `relative_risk`, `risk_difference` intervals|Use a hard-coded z = 1.96 rather than the exact 0.975 quantile 1.959964, moving the bounds by about 1.3e-05 relative.|
|`pca`|Obtains components by power iteration with deflation, agreeing with `prcomp` to about 1.8e-07; scores amplify that to 1.8e-06. Eigenvector signs are arbitrary and are normalised on both sides.|
|`mahalanobis_distance`|Supports two dimensions only (documented) and returns the distance, not the squared distance R returns.|
|logistic `coefficient_se`|Agrees with R to 3.2e-05 only: the IRLS stops on the deviance, which is flat at the optimum, so coefficients settle near sqrt(machine epsilon) and inverting X'WX amplifies that. Poisson reaches 1.1e-10.|
|`norm_cdf`, `normal_cdf`|Resolved. Previously evaluated as `0.5 (1 + erf(x / sqrt(2)))`, which cancelled catastrophically in the left tail: the result was exactly zero below x = -8.327. Now evaluated as `0.5 erfc(-x / sqrt(2))`, which agrees with R to 1.9e-13 relative down to x = -37.5 and also improves the centre from 2.2e-14 to 1.7e-15.|
|`normal_quantile`|Agrees with R to 7.1e-08 relative, one tier looser than the 1e-8 default.|
|`studentized_range_quantile`|Inverted by root finding; agrees with R to 4.3e-07 relative.|
|`gammainc_upper`|Far upper tail computed by continued fraction; agrees with R to 3.1e-07 relative.|
