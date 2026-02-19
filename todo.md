# C++ Statistics Package Development Project

This document arranges blocks in development-appropriate order, considering implementation dependencies.

---

## Module 1: Descriptive Statistics

Pure statistical functions operating on iterator pairs. Foundation for other features.

### Basic Statistics

- [x] Sum
- [x] Count / N
- [x] Mean — Arithmetic Mean
- [x] Median
- [x] Mode
- [x] Geometric Mean
- [x] Harmonic Mean
- [x] Trimmed Mean
- [x] Weighted Mean
- [x] Logarithmic Mean
- [x] Weighted Harmonic Mean

### Dispersion & Spread

- [x] Range — Maximum - Minimum
- [x] Variance — Population / Sample Variance
- [x] Standard Deviation — Population / Sample Standard Deviation
- [x] Coefficient of Variation
- [x] Interquartile Range (IQR)
- [x] Mean Absolute Deviation
- [ ] Variance ddof parameter
- [x] Weighted Variance
- [x] Geometric Standard Deviation
- [ ] Log Standard Deviation

### Order Statistics

- [x] Minimum
- [x] Maximum
- [x] Quartiles — Q1, Q2, Q3
- [x] Percentiles
- [x] Five-Number Summary
- [x] argmin / argmax (index retrieval)
- [ ] Quantile interpolation methods
- [x] Weighted Percentiles
- [x] Weighted Median
- [ ] Deciles
- [ ] Percentile Rank

### Shape of Distribution

- [x] Skewness
- [x] Kurtosis
- [ ] Bias-corrected Skewness/Kurtosis
- [ ] Central Moments
- [ ] Higher-order Statistics

### Correlation & Covariance

- [x] Covariance
- [x] Pearson Correlation Coefficient
- [x] Spearman's Rank Correlation
- [x] Kendall's Tau
- [ ] Partial Correlation
- [x] Weighted Covariance
- [ ] Weighted Correlation Matrix

### Frequency Distribution

- [x] Frequency Table
- [x] Relative Frequency
- [x] Cumulative Frequency
- [x] Cumulative Relative Frequency
- [ ] Histogram Computation
- [ ] Binned Statistics
- [ ] Automatic Binning (Scott/Freedman-Diaconis)

### Utility Functions

- [ ] NaN-aware Aggregations
- [ ] Partial Selection (destructive/non-destructive)
- [ ] Kernel Density Estimation (KDE)

**Module 1 Unimplemented Items Rationale**:
- **Variance ddof parameter**: Already covered by `population_variance` and `sample_variance`. Additional API is redundant.
- **Log Standard Deviation**: Similar functionality provided by geometric standard deviation. Low usage frequency.
- **Quantile interpolation methods**: Current linear interpolation (R type=7 equivalent) is sufficient. Multiple interpolation methods would complicate API.
- **Deciles, Percentile Rank**: Achievable with `percentile` function. Dedicated functions unnecessary.
- **Bias-corrected Skewness/Kurtosis, Central Moments, Higher-order Statistics**: Low usage frequency, low implementation priority.
- **Partial Correlation**: Can be substituted with linear regression residual correlation. Dedicated implementation is complex.
- **Weighted Correlation Matrix**: Can be constructed from individual weighted covariances.
- **Histogram Computation, Binned Statistics, Automatic Binning**: Visualization-oriented features. Outside scope of header-only library.
- **NaN-aware Aggregations**: C++ design favors explicit NaN checking on user side.
- **Partial Selection**: Can use `std::nth_element` and other standard algorithms.
- **Kernel Density Estimation (KDE)**: Complex algorithm. External library recommended.

---

## Module 2: Probability Distributions

Foundation for tests and estimation. For each distribution, implement pdf/pmf, cdf, quantile, rng.

### Development Foundation: Special Functions

Prepare special functions needed for probability distribution implementation.

- [x] Gamma / Log-Gamma Function
- [x] Beta / Incomplete Beta Function
- [x] Error Function (erf/erfc)
- [x] Normal CDF & Quantile
- [ ] Additional Special Functions

### Random Number Generation

- [x] RNG Engine (Seed & Reproducibility)

### Continuous Distributions

Implementation order: Prioritize basic distributions and those that serve as foundations for others

1. - [x] Uniform Distribution — pdf / cdf / quantile / rng
2. - [x] Normal Distribution — pdf / cdf / quantile / rng
3. - [x] Exponential Distribution — pdf / cdf / quantile / rng
4. - [x] Gamma Distribution — pdf / cdf / quantile / rng
5. - [x] Beta Distribution — pdf / cdf / quantile / rng
6. - [x] Chi-square Distribution — pdf / cdf / quantile / rng ※Special case of Gamma
7. - [x] t-Distribution — pdf / cdf / quantile / rng
8. - [x] F-Distribution — pdf / cdf / quantile / rng
9. - [x] Log-normal Distribution — pdf / cdf / quantile / rng
10. - [x] Weibull Distribution — pdf / cdf / quantile / rng
11. - [ ] Logistic Distribution — pdf / cdf / quantile / rng
12. - [ ] Laplace Distribution — pdf / cdf / quantile / rng
13. - [ ] Cauchy Distribution — pdf / cdf / quantile / rng
14. - [ ] Pareto Distribution — pdf / cdf / quantile / rng
15. - [ ] Inverse Gamma Distribution — pdf / cdf / quantile / rng
16. - [ ] Gumbel Distribution (Extreme Value Type I) — pdf / cdf / quantile / rng
17. - [ ] Triangular Distribution — pdf / cdf / quantile / rng
18. - [ ] Erlang Distribution — pdf / cdf / quantile / rng

### Discrete Distributions

1. - [x] Binomial Distribution — pmf / cdf / quantile / rng
2. - [x] Poisson Distribution — pmf / cdf / quantile / rng
3. - [x] Geometric Distribution — pmf / cdf / quantile / rng
4. - [x] Negative Binomial Distribution — pmf / cdf / quantile / rng
5. - [x] Hypergeometric Distribution — pmf / cdf / quantile / rng
6. - [x] Bernoulli Distribution — pmf / cdf / quantile / rng
7. - [ ] Categorical Distribution — pmf / cdf / quantile / rng
8. - [ ] Multinomial Distribution — pmf / cdf / quantile / rng
9. - [x] Discrete Uniform Distribution — pmf / cdf / quantile / rng

### Multivariate Distributions

- [ ] Multivariate Normal Distribution
- [ ] Multivariate Student's t Distribution
- [ ] Dirichlet Distribution

### Distribution Extensions

- [ ] Log-likelihood APIs (logpdf/logpmf)
- [ ] Survival Function
- [ ] Hazard Function
- [ ] Theoretical Moments (mean, variance, skewness, kurtosis)
- [ ] Parameter Estimation for Distributions
- [ ] Empirical Distribution
- [ ] Dirac Delta Distribution

**Module 2 Unimplemented Items Rationale**:
- **Additional continuous distributions (Logistic, Laplace, Cauchy, etc.)**: Major distributions implemented. Additional distributions have limited use cases.
- **Categorical Distribution, Multinomial Distribution**: Requires multi-dimensional array handling, complex for header-only. Can be substituted with Bernoulli/Binomial distributions.
- **Multivariate Distributions**: Requires linear algebra library (matrix operations, Cholesky decomposition, etc.). Significantly exceeds header-only scope.
- **Log-likelihood API**: Easily achieved with `std::log(pdf(x))`. Dedicated API is redundant.
- **Survival Function, Hazard Function**: Implemented in survival analysis (Kaplan-Meier, etc.). Generic distribution function implementation is low priority.
- **Theoretical Moments**: Mean/variance formulas for each distribution are well-known. Limited value in functionalization.
- **Parameter Estimation**: MLE, method of moments require complex optimization. Module 3-4 scope.
- **Empirical Distribution**: Can be computed directly from data. Dedicated distribution class unnecessary.
- **Dirac Delta Distribution**: Theoretical concept. Low practical use in numerical computation.
- **Additional Special Functions**: Currently implemented special functions cover major distributions.

---

## Module 3: Inferential Statistics

Depends on probability distributions. Statistical tests and estimation.

### Estimation

- [x] Standard Error Estimation
- [x] Confidence Intervals (Mean/Proportion/Variance, etc.)
- [x] Mean Margin of Error
- [x] Proportion Margin of Error
- [x] Worst-case Proportion Margin of Error
- [x] Two-sample Mean Difference
- [x] Two-sample Proportion Difference
- [ ] Maximum Likelihood Estimation (MLE)
- [ ] Method of Moments

### Parametric Tests

- [x] z-tests (Mean/Proportion)
- [x] t-tests (1-sample/2-sample/paired)
- [x] Chi-square Tests (GOF/Independence)
- [x] F-tests (Variance Comparison, etc.)
- [x] Proportion Tests
- [x] One-sided/Two-sided Testing
- [x] Multiple Testing Correction (Bonferroni, BH, etc.)
- [ ] Additional Multiple Testing Corrections

### Nonparametric Tests

- [x] Normality Tests (Shapiro-Wilk, Anderson-Darling, KS, etc.)
- [x] Homoscedasticity Tests (Levene, Bartlett, etc.)
- [x] Wilcoxon Signed-rank Test
- [x] Mann-Whitney U Test
- [x] Kruskal-Wallis Test
- [x] Rank Correlations (Spearman/Kendall)
- [x] Fisher's Exact Test

### Effect Size & Power

- [x] Effect Sizes (Cohen's d, Hedges' g, correlation effect sizes, etc.)
- [x] Power Analysis & Sample Size Planning

### Resampling

- [x] Bootstrap (CIs/Bias Correction)
- [x] Permutation Tests
- [x] Cross-validation (Core Utilities)
- [ ] Additional Resampling Methods

**Module 3 Unimplemented Items Rationale**:
- **Maximum Likelihood Estimation (MLE), Method of Moments**: Requires complex optimization algorithms. Partially covered in Module 4 modeling (GLM, etc.). Generic MLE implementation is out of scope.
- **Additional Multiple Testing Corrections**: Bonferroni, Benjamini-Hochberg cover major cases. Additional methods have low usage frequency.
- **Additional Resampling Methods**: Bootstrap, permutation tests, cross-validation cover major methods. Jackknife etc. can be substituted with similar concepts.

---

## Module 4: Statistical Modeling

Depends on inferential statistics. Regression, GLM, ANOVA.

### Linear Regression / OLS

- [x] Least Squares Estimation
- [x] Coefficient SEs & Tests
- [x] R-squared
- [x] Residual Diagnostics
- [x] Multicollinearity (VIF, etc.)
- [x] Prediction Intervals
- [ ] Robust Regression (Huber, RANSAC, etc.)

### ANOVA & Extensions

- [x] One-way/Two-way ANOVA
- [x] ANCOVA
- [x] Interaction Effects
- [x] Post-hoc Comparisons (Tukey, Dunnett, etc.)

### Generalized Linear Models / GLM

- [x] Logistic Regression
- [x] Poisson Regression
- [x] Link & Variance Functions
- [x] Iterative Optimization (IRLS, etc.)
- [x] Deviance & Goodness-of-fit Metrics
- [ ] Extended Logistic Regression (regularization, etc.)

### Model Selection & Regularization

- [x] Model Selection Criteria (AIC/BIC/Adjusted R²)
- [x] Regularized Regression (Ridge/Lasso/Elastic Net)
- [x] Cross-validation Variants (k-fold/stratified/time series CV)

### Advanced Modeling

- [ ] Mixed-effects Models (LMM/GLMM)
- [ ] Generalized Additive Models (GAM)
- [x] Extended Multicollinearity Diagnostics (condition number, etc.)

### Bayesian Inference ※Optional

- [ ] Conjugate Priors (Basics)
- [ ] MCMC (Introduction)
- [ ] Variational Inference (Optional)
- [ ] Model Comparison
- [ ] Posterior Probability / Bayes Factor

**Module 4 Unimplemented Items Rationale**:
- **Robust Regression (Huber, RANSAC, etc.)**: Requires iterative optimization and weight recalculation. Complex implementation, recommend external library (e.g., Ceres Solver).
- **Extended Logistic Regression (regularization, etc.)**: Regularization concepts implemented in Ridge/Lasso regression. Application to logistic regression possible with similar implementation, but low priority.
- **Mixed-effects Models (LMM/GLMM)**: Random effects estimation requires complex matrix operations and optimization. Recommend specialized libraries (lme4, nlme, etc.).
- **Generalized Additive Models (GAM)**: Requires spline function implementation and smoothing parameter selection. Very complex.
- **Bayesian Inference (MCMC, Variational Inference, etc.)**: Requires large-scale numerical computation infrastructure. Should be delegated to specialized tools like Stan, PyMC. Conjugate prior-only implementation has limited practical use.

---

## Module 5: Applied & Domain-Specific Analysis

Analysis methods tied to specific data structures or domains. Leverages Module 1-4 features.

### Multivariate Analysis

- [x] Covariance/Correlation Matrix
- [x] Principal Component Analysis (PCA)
- [ ] Factor Analysis
- [ ] Discriminant Analysis (Optional)
- [x] Standardization/Scaling Integration
- [ ] Extended Standardization/Scoring

### Time Series Analysis

- [x] ACF/PACF
- [ ] ARIMA (Basics)
- [ ] Seasonality Handling
- [x] Forecast Evaluation (MAE/RMSE/MAPE, etc.)
- [ ] Time Series Decomposition (STL, etc.)
- [ ] Extended ARIMA/Seasonality

### Categorical Data Analysis

- [x] Contingency Tables
- [x] Odds Ratio / Relative Risk
- [ ] Log-linear Models (Optional)

### Survival Analysis

- [x] Kaplan-Meier Estimator
- [x] Log-rank Test
- [ ] Cox Proportional Hazards Model
- [ ] Extended Survival Analysis

### Robust Statistics & Diagnostics

- [x] Median Absolute Deviation (MAD)
- [x] Robust Estimators Extensions
- [ ] Robust Regression (Optional)
- [x] Influence Measures (Cook's Distance, etc.)
- [x] Outlier Detection (IQR/Tukey, LOF, etc.)
- [ ] Extended Trimming/Outlier Removal

### Clustering & Dimensionality Reduction

- [x] Clustering (k-means, hierarchical, DBSCAN, etc.)
- [ ] Dimensionality Reduction (t-SNE/UMAP; Optional)

### Distance & Similarity Metrics

- [x] Euclidean Distance
- [x] Manhattan Distance
- [x] Mahalanobis Distance
- [x] Cosine Similarity
- [x] Other Statistical Distances (Minkowski, Chebyshev)

### Information Theory

- [ ] Entropy
- [ ] Mutual Information
- [ ] Kullback-Leibler Divergence
- [ ] Jensen-Shannon Divergence
- [ ] Information-theoretic Differences

### Directional Statistics

- [ ] Circular Mean
- [ ] Circular Variance
- [ ] von Mises Distribution

**Module 5 Unimplemented Items Rationale**:
- **Factor Analysis**: Requires eigenvalue decomposition and factor rotation. Heavy dependency on linear algebra library. PCA provides similar dimensionality reduction.
- **Discriminant Analysis**: Linear Discriminant Analysis (LDA) requires linear algebra. Can be substituted with logistic regression.
- **Extended Standardization/Scoring**: Basic standardization implemented. Additional scaling methods (Robust Scaler, etc.) are low priority.
- **ARIMA (Basics), Seasonality Handling, Time Series Decomposition**: Very complex algorithms. Recommend specialized libraries (statsmodels, forecast, etc.).
- **Log-linear Models**: Can be implemented as a type of GLM, but categorical data handling is complex. Limited usage frequency.
- **Cox Proportional Hazards Model**: Requires partial likelihood optimization. Basic survival analysis (Kaplan-Meier, log-rank test) implemented. Cox regression is too specialized.
- **Extended Survival Analysis**: Kaplan-Meier estimator and log-rank test enable basic analysis. Extensions like competing risks analysis are specialized.
- **Robust Regression**: Same reason as Module 4. Requires iterative optimization, complex.
- **Extended Trimming/Outlier Removal**: Basic outlier detection (IQR method, MAD, etc.) implemented. Additional methods are low priority.
- **Dimensionality Reduction (t-SNE/UMAP)**: Non-linear dimensionality reduction is very complex. Requires iterative optimization and neighbor search. Recommend specialized libraries.
- **Entropy, Mutual Information, KL Divergence, JS Divergence**: Implementable for discrete probability distributions, but continuous distributions require numerical integration. Medium priority, but distance metrics prioritized for now.
- **Information-theoretic Differences**: Basic information theory quantities not implemented, so derived features also unimplemented.
- **Directional Statistics (Circular Mean, Circular Variance, von Mises Distribution)**: For specialized domain (angle data). Very limited usage frequency.

---

## Module 6: Data Infrastructure

### Data Structures

- [ ] Data Container Design (Series / DataFrame equivalent)
- [ ] Type System (numeric/categorical/datetime/missing)
- [ ] Type Inference & Schema Definition
- [ ] Indexing & Labels (row/column names)

### I/O

- [ ] CSV/TSV I/O
- [ ] JSON I/O
- [ ] Columnar/Binary Formats (Parquet, etc.)

### Data Wrangling & Preprocessing

- [x] Missing Data Handling (Drop/Impute)
- [x] Outlier Handling (Detection/Removal/Winsorize) ※Implemented in robust.hpp
- [x] Filtering
- [x] Transformations & Derived Columns
- [x] Group-by & Aggregation
- [ ] Joins / Merges
- [ ] Reshaping (Wide/Long)
- [x] Sorting (single/composite key, stable sort)
- [x] Sampling (random/stratified/with or without replacement)
- [x] Duplicate Detection & Deduplication
- [x] Rolling/Window Aggregations (moving average, moving variance, etc.)
- [x] Categorical Encoding (one-hot/ordinal, etc.)
- [x] Data Validation (range/type/missing rate, etc.)

### Advanced Missing Data

- [x] MCAR/MAR/MNAR Taxonomy
- [x] Multiple Imputation
- [x] Sensitivity Analysis

**Module 6 Unimplemented Items Rationale**:
Can be developed independently of statistical functions. Requires large-scale design, to be started after core feature completion.

---

## Module 7: Visualization

### EDA Visualization

- [ ] Histogram (bin selection: Scott/Freedman-Diaconis, etc.)
- [ ] Box Plot
- [ ] Q-Q Plot
- [ ] Scatter Plot
- [ ] Scatterplot Matrix
- [ ] Correlation Heatmap
- [ ] Category-wise Comparison (bar/violin, etc.)

### Diagnostic Plots

- [ ] Regression Diagnostic Plots (residuals vs fitted, QQ, influence)
- [ ] ACF/PACF Plots
- [ ] ROC Curve & AUC (if classification models are included)

**Module 7 Unimplemented Items Rationale**:
To be started after other features are complete. Integration with external libraries also under consideration.

---

## Module 8: Development Infrastructure

To be developed in parallel with each Module.

### Numerical & Optimization Core

- [ ] Optimization (gradient, Hessian, convergence criteria)
- [ ] Linear Algebra (decomposition, stabilization)
- [x] Numerical Stability & Precision Design
- [ ] Numerical Integration (for CDF/expectation calculation)
- [ ] Numerical/Automatic Differentiation (Optional)
- [ ] Performance Optimization (SIMD/parallelization consideration)
- [x] Precision & Convergence Utilities

### Reproducibility & Reporting

- [ ] RNG & Environment Capture
- [ ] Analysis Logging (parameters, version)
- [ ] Result Object Serialization
- [ ] Formatted Output for Tables/Reports

### API Design & Interoperability

- [ ] Pipeline/Chaining API
- [ ] Error Model & Exception Design
- [ ] Extensibility (Plugin Architecture)
- [ ] Language Bindings (Optional)

### Testing & Benchmarking

- [ ] Testing (known values, boundary values, random tests)
- [ ] Benchmarking (performance comparison by algorithm)

**Module 8 Unimplemented Items Rationale**:

**Numerical & Optimization Core**:
- **Optimization (gradient, Hessian, convergence criteria)**: Complex optimization algorithms (BFGS, L-BFGS, etc.) recommend external libraries (Ceres, NLopt, etc.). Generic implementation significantly exceeds header-only scope.
- **Linear Algebra (decomposition, stabilization)**: LU decomposition, QR decomposition, SVD, etc. require large-scale implementation. Recommend specialized libraries like Eigen, Armadillo.
- **Numerical Integration**: Major probability distribution CDFs implemented with existing special functions (incomplete beta, gamma functions, etc.). Additional generic numerical integration is low priority.
- **Numerical/Automatic Differentiation**: Finite difference method is easily implemented but low precision. Automatic differentiation possible with template metaprogramming but very complex. Recommend external libraries (autodiff, CppAD, etc.).
- **Performance Optimization (SIMD/parallelization)**: Requires compiler-dependent, platform-dependent implementation. Maintainability decreases for header-only library. Policy to delegate optimization to compiler.

**Reproducibility & Reporting**:
- **RNG & Environment Capture**: Random seed setting possible in `random_engine.hpp`. Environment information (OS, compiler version, etc.) capture is difficult in header-only.
- **Analysis Logging, Result Serialization**: Requires file I/O. Outside scope of header-only library. Should be implemented on user side.
- **Formatted Output for Tables/Reports**: Statistical result display format highly depends on user needs. More flexible to return result structs and let users format freely rather than forcing standard format.

**API Design & Interoperability**:
- **Pipeline/Chaining API**: Would require significant changes to current iterator-based API. Loses compatibility with existing design. Should wait for Ranges library (C++20+) adoption.
- **Error Model & Exception Design**: Exception handling with `std::invalid_argument` uniformly implemented in all functions. No additional design needed.
- **Extensibility (Plugin Architecture)**: Plugin mechanism implementation is difficult in header-only library. Users can implement custom analysis by combining existing functions.
- **Language Bindings**: Bindings for Python (pybind11), R (Rcpp), etc. should be developed as separate projects. statcpp itself is complete as a C++ library.

**Testing & Benchmarking**:
- **Testing (known values, boundary values, random tests)**: Test code created for each feature in Modules 1-5 and 8. Comprehensively tests known values, boundary values, special values (NaN, infinity, etc.). Additional systematic test framework is low priority.
- **Benchmarking (performance comparison by algorithm)**: Benchmarks depend on user environment and data. Limited value in providing as library. Should be conducted by users as needed.

---

## Implementation Notes

### Priority Considerations

1. **Phase 1-3**: Statistical fundamentals. Highest priority implementation
2. **Phase 4**: Modeling. Depends on Phase 1-3
3. **Phase 5**: Applied analysis. Domain-specific
4. **Phase 6-7**: Data infrastructure and visualization. Requires large-scale design
5. **Phase 8**: Development infrastructure. Continuously maintained throughout

### Positioning of Additional Items in Each Phase

- **Weighted Statistics**: Added to Phase 1 (natural extension of basic statistics)
- **Additional Probability Distributions**: Added to Phase 2 (distribution enrichment)
- **Information Theory/Distance**: New section in Phase 5 (applied)
- **Directional Statistics**: New section in Phase 5 (specialized domain)
- **Numerical Computation Infrastructure Extension**: Integrated into Phase 8

### Implemented Items

Phase 1-5 major features implementation complete. Currently in sample code creation phase.
