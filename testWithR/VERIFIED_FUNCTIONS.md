# Functions Verified Against R

List of functions verified against R 4.4.2 output in `verify_vs_r.cpp`.

**Summary: 57 functions directly verified, 167 numerical checks, all PASS.**

## parametric_tests.hpp

| statcpp Function | R Function | Verified Items |
|-----------------|------------|----------------|
| `t_test()` | `t.test(x, mu=...)` | t-statistic, df, p-value |
| `t_test_two_sample()` | `t.test(x, y, var.equal=TRUE)` | t-statistic, df, p-value |
| `t_test_welch()` | `t.test(x, y)` | t-statistic, df, p-value |
| `t_test_paired()` | `t.test(x, y, paired=TRUE)` | t-statistic, df, p-value |
| `f_test()` | `var.test(x, y)` | F-statistic, p-value |
| `chisq_test_gof()` | `chisq.test(obs, p=...)` | X-squared, df, p-value |
| `chisq_test_independence()` | `chisq.test(matrix, correct=FALSE)` | X-squared, df, p-value |
| `z_test_proportion()` | `prop.test(x, n, p, correct=FALSE)` | z-squared = X-squared, p-value |
| `z_test_proportion_two_sample()` | `prop.test(c(x1,x2), c(n1,n2), correct=FALSE)` | z-squared = X-squared, p-value |
| `bonferroni_correction()` | `p.adjust(p, method="bonferroni")` | All corrected p-values |
| `benjamini_hochberg_correction()` | `p.adjust(p, method="BH")` | All corrected p-values |

## nonparametric_tests.hpp

| statcpp Function | R Function | Verified Items |
|-----------------|------------|----------------|
| `wilcoxon_signed_rank_test()` | `wilcox.test(x, mu=..., correct=TRUE)` | V-statistic, p-value |
| `mann_whitney_u_test()` | `wilcox.test(x, y)` | U-statistic, p-value |
| `kruskal_wallis_test()` | `kruskal.test()` | H-statistic, df, p-value |
| `shapiro_wilk_test()` | `shapiro.test()` | W-statistic, p-value |
| `levene_test()` | Median-based ANOVA on abs deviations | F-statistic, p-value |
| `bartlett_test()` | `bartlett.test()` | K-squared, p-value |
| `fisher_exact_test()` | `fisher.test()` | p-value |

## anova.hpp

| statcpp Function | R Function | Verified Items |
|-----------------|------------|----------------|
| `one_way_anova()` | `summary(aov(...))` | F-statistic, p-value, MSb, MSw |
| `tukey_hsd()` | `TukeyHSD(aov(...))` | \|diff\|, p-value, CI width |

## effect_size.hpp

| statcpp Function | R Function | Verified Items |
|-----------------|------------|----------------|
| `cohens_d_two_sample()` | Manual calculation | Cohen's d |
| `hedges_g_two_sample()` | Manual calculation (J correction) | Hedges' g |
| `eta_squared()` (standalone) | Manual calculation | Eta-squared |
| `eta_squared()` (ANOVA) | Equivalent to `effectsize::eta_squared()` | Eta-squared |
| `omega_squared()` | Equivalent to `effectsize::omega_squared()` | Omega-squared |
| `cohens_f()` | Equivalent to `effectsize::cohens_f()` | Cohen's f |

## linear_regression.hpp

| statcpp Function | R Function | Verified Items |
|-----------------|------------|----------------|
| `simple_linear_regression()` | `lm(y ~ x)` | Intercept, slope, R-squared, SE(b0), SE(b1) |
| `multiple_linear_regression()` | `lm(Y ~ X1 + X2)` | Coefficients (b0, b1, b2), R-squared |

## glm.hpp

| statcpp Function | R Function | Verified Items |
|-----------------|------------|----------------|
| `logistic_regression()` | `glm(y~x, family=binomial)` | Coefficients, deviance, null deviance, AIC |
| `poisson_regression()` | `glm(y~x, family=poisson)` | Coefficients, deviance, AIC |

## estimation.hpp

| statcpp Function | R Function | Verified Items |
|-----------------|------------|----------------|
| `ci_mean()` | `t.test(x, conf.level=0.95)` | Lower bound, upper bound |

## categorical.hpp

| statcpp Function | R Function | Verified Items |
|-----------------|------------|----------------|
| `odds_ratio()` | `(a*d)/(b*c)` + `exp(log(OR) +/- 1.96*SE)` | OR, CI lower, CI upper |
| `relative_risk()` | Manual calculation | RR, CI lower, CI upper |
| `risk_difference()` | Manual calculation | RD, CI lower, CI upper |
| `number_needed_to_treat()` | `1/RD` | NNT |

## shape_of_distribution.hpp

| statcpp Function | R Function | Verified Items |
|-----------------|------------|----------------|
| `population_skewness()` | `e1071::skewness(x, type=1)` | Skewness |
| `sample_skewness()` | `e1071::skewness(x, type=2)` | Skewness |
| `population_kurtosis()` | `e1071::kurtosis(x, type=1)` | Excess kurtosis |

## survival.hpp

| statcpp Function | R Function | Verified Items |
|-----------------|------------|----------------|
| `kaplan_meier()` | `survfit(Surv(t,s) ~ 1)` | Survival probability (6 time points), SE (3 time points) |
| `median_survival_time()` | `survfit()$median` | Median survival time |
| `nelson_aalen()` | Manual calculation (cumhaz) | Cumulative hazard (6 time points) |
| `logrank_test()` | `survdiff(Surv(t,s) ~ g)` | Chi-squared, p-value, observed counts, expected counts |

## model_selection.hpp

| statcpp Function | R Function | Verified Items |
|-----------------|------------|----------------|
| `aic()` | `-2*logL + 2*k` | AIC value |
| `bic()` | `-2*logL + k*log(n)` | BIC value |
| `aicc()` | `AIC + 2k(k+1)/(n-k-1)` | AICc value |

## power_analysis.hpp

| statcpp Function | R Function | Verified Items |
|-----------------|------------|----------------|
| `power_t_test_one_sample()` | Range check | 0.5 < power < 1.0 |
| `power_t_test_two_sample()` | Range check | 0.3 < power < 1.0 |
| `sample_size_t_test_one_sample()` | Range check | 25 <= n <= 45 |
| `sample_size_t_test_two_sample()` | Range check | 50 <= n <= 80 |

**Note**: Power analysis is verified by range checks (output within reasonable bounds) rather than exact comparison with R's `pwr` package.

## robust.hpp

| statcpp Function | R Function | Verified Items |
|-----------------|------------|----------------|
| `mad()` | `mad(x, constant=1)` | MAD (unscaled) |
| `mad_scaled()` | `mad(x)` | MAD (x 1.4826) |
| `winsorize()` | Quantile-based manual calculation | Endpoints and interior values |

## time_series.hpp

| statcpp Function | R Function | Verified Items |
|-----------------|------------|----------------|
| `acf()` | `acf(y, lag.max=3)` | Autocorrelations at lags 0--3 |
| `pacf()` | `pacf(y, lag.max=3)` | Partial autocorrelations at lags 0--3 |
| `diff()` | `diff(y)` | Differenced values |
| `mae()` | `Metrics::mae()` | MAE |
| `mse()` | `Metrics::mse()` | MSE |
| `rmse()` | `Metrics::rmse()` | RMSE |

## multivariate.hpp

| statcpp Function | R Function | Verified Items |
|-----------------|------------|----------------|
| `covariance_matrix()` | `cov(X)` | Covariance matrix (6 elements) |
| `correlation_matrix()` | `cor(X)` | Correlation matrix (4 elements) |

## data_wrangling.hpp

| statcpp Function | R Function | Verified Items |
|-----------------|------------|----------------|
| `log_transform()` | `log(x)` | Log-transformed values |
| `sqrt_transform()` | `sqrt(x)` | Square-root-transformed values |

## continuous_distributions.hpp

**Note**: `verify_vs_r.cpp` does not directly test continuous distribution functions, but they are indirectly verified through parametric tests (t-test, F-test, chi-squared test, etc.) whose p-values match R. Agreement in p-values implies that the internally used `norm_cdf`, `t_cdf`, `f_cdf`, `chisq_cdf`, etc. are correct.

Similarly, `studentized_range_cdf` / `studentized_range_quantile` are verified through the Tukey HSD tests.

## discrete_distributions.hpp

**Note**: The hypergeometric distribution functions used internally are indirectly verified through the exact p-value computation in `fisher_exact_test`.
