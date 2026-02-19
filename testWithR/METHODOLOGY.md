# Cross-Verification with R — Methodology

## Overview

`verify_vs_r.cpp` is a standalone program that compares the output of each statcpp function against reference values from R (version 4.4.2), verifying numerical agreement. It does not depend on Google Test.

## Verification Environment

| Item | Value |
|------|-------|
| R version | R 4.4.2 |
| C++ standard | C++17 |
| Default tolerance | 1e-6 (absolute error) |
| Functions directly verified | 57 |
| Numerical checks | 167 |
| Test results | **167 PASS, 0 FAIL** |

## How the Verification Program Works

### The check() Function

```cpp
static void check(const char* name, double cpp_val, double r_val, double tol = 1e-6);
```

- **Normal values**: PASS if `|cpp_val - r_val| < tol`
- **Infinity**: PASS if both are infinity with the same sign
- **NaN**: PASS if both are NaN
- **On failure**: Prints the function name, C++ value, R value, and the difference

### Tolerance Settings

Most tests use the default `tol = 1e-6`. The following cases use explicitly larger tolerances:

| Function | Tolerance | Reason |
|----------|-----------|--------|
| GLM (logistic/poisson) | 0.1 -- 1.0 | Differences in IRLS optimization convergence paths |
| Shapiro-Wilk p-value | 0.05 | Differences in approximation algorithms |
| Log-rank statistic | 0.1 | Rounding differences in expected value computation |
| PACF (lag 3) | 0.05 | Differences in Yule-Walker solver implementations |

## How R Reference Values Were Generated

R reference values for each test were generated as follows:

1. Define the same data in an R script
2. Run the corresponding R function
3. Output with sufficient precision using `sprintf("%.15g", result)`
4. Hard-code the output values into `check()` calls in `verify_vs_r.cpp`

### Example R Code

```r
# One-sample t-test
x <- c(5.1, 4.9, 5.3, 5.1, 4.8, 5.2, 5.0, 4.9)
r <- t.test(x, mu=5.0)
sprintf("%.15g", r$statistic)   # t-statistic
sprintf("%.15g", r$p.value)     # p-value
```

## Convention Differences Between C++ and R

The following convention differences between C++ and R were discovered during verification:

### 1. Tukey HSD Difference Sign

| | C++ (statcpp) | R (TukeyHSD) |
|---|---|---|
| Comparison direction | group_i - group_j | group_j - group_i |
| Sign of difference | Positive (when i < j and group i has larger mean) | Negative |

Tests compare using `|diff|` and verify correctness through CI width.

### 2. PACF Indexing

| | C++ (statcpp) | R (pacf) |
|---|---|---|
| index [0] | lag 0 = 1.0 | lag 1 value |
| index [1] | lag 1 value | lag 2 value |

C++ includes lag 0; R starts from lag 1. Tests adjust indices accordingly.

### 3. Wilcoxon Test Continuity Correction

| | C++ (statcpp) | R default |
|---|---|---|
| Continuity correction | Applied | `correct=TRUE` (default) |

Both apply continuity correction. Tests use R's `correct=TRUE` reference values.

### 4. Winsorize Quantile Calculation

| | C++ (statcpp) | R (DescTools::Winsorize) |
|---|---|---|
| Quantile method | type 7 (R's quantile default) | type 7 |

Same method. Tests use `quantile(x, 0.10)` / `quantile(x, 0.90)` values.

## Test Sections

| Section | # Tests | Key Verifications |
|---------|---------|-------------------|
| PARAMETRIC TESTS | 22 | t-tests (one-sample/two-sample/Welch/paired), F-test, chi-squared test, z-proportion test, multiple testing corrections |
| NONPARAMETRIC TESTS | 16 | Wilcoxon signed-rank, Mann-Whitney U, Kruskal-Wallis, Shapiro-Wilk, Levene, Bartlett, Fisher exact test |
| ANOVA | 10 | One-way ANOVA (F/p/MS), Tukey HSD, effect sizes (eta-squared/omega-squared/Cohen's f) |
| LINEAR REGRESSION | 8 | Simple regression (coefficients/R-squared/SE), multiple regression (coefficients/R-squared) |
| GLM | 10 | Logistic regression, Poisson regression (coefficients/deviance/AIC) |
| EFFECT SIZE | 3 | Cohen's d, Hedges' g, eta-squared |
| ESTIMATION (CI) | 2 | Confidence interval for the mean |
| CATEGORICAL | 8 | Odds ratio, relative risk, risk difference (each with CI), NNT |
| SHAPE OF DISTRIBUTION | 3 | Population skewness, sample skewness, population kurtosis |
| SURVIVAL | 14 | Kaplan-Meier (survival probability/SE/median survival time), Nelson-Aalen, log-rank test |
| MODEL SELECTION | 3 | AIC, BIC, AICc |
| POWER ANALYSIS | 4 | Power (one-sample/two-sample), required sample size (range checks) |
| ROBUST STATISTICS | 6 | MAD (scaled/unscaled), Winsorize |
| TIME SERIES | 12 | ACF, PACF, differencing, MAE/MSE/RMSE |
| MULTIVARIATE | 8 | Covariance matrix, correlation matrix |
| DATA WRANGLING | 4 | Log transform, square root transform |
| **Total** | **167** | |

## Build and Run

```bash
# EN build
cmake -B build -S . && cmake --build build
./build/testWithR/verify_vs_r

# JA build
cmake -B build-ja -S . -DSTATCPP_USE_JAPANESE=ON && cmake --build build-ja
./build-ja/testWithR/verify_vs_r
```

## Notes

- **Power Analysis** is verified by range checks (i.e., output falls within a reasonable range) rather than exact comparison with R's `pwr` package.
- **GLM** uses larger tolerances due to implementation differences in IRLS algorithms.
- All tests feed the same dataset into both C++ and R and compare outputs.
