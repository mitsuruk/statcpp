# Changelog

This document records the change history of the statcpp library.

This project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned Features

- Support for more distribution functions
- Performance optimizations
- More detailed error messages

---

## [0.1.0] - 2024-02-02

### Initial Release

#### Added Features

**Basic Statistics**
- Sum, mean, median, mode
- Geometric mean, harmonic mean, trimmed mean
- Weighted mean, logarithmic mean

**Dispersion & Spread**
- Variance, standard deviation, range
- Coefficient of variation, interquartile range
- Mean absolute deviation

**Order Statistics**
- Minimum, maximum
- Quartiles, percentiles
- Five-number summary

**Shape of Distribution**
- Skewness
- Kurtosis, excess kurtosis

**Correlation & Covariance**
- Pearson correlation coefficient
- Spearman rank correlation coefficient
- Kendall's rank correlation coefficient
- Covariance

**Frequency Distribution**
- Frequency distribution table
- Histogram
- Cumulative frequency distribution

**Special Functions**
- Gamma function, beta function
- Error function
- Normal distribution CDF, PDF, inverse function

**Random Number Generation**
- Uniform distribution, normal distribution random generation

**Probability Distributions**
- Continuous distributions: Normal, t, chi-squared, F, exponential, gamma, beta, Weibull, log-normal, Cauchy
- Discrete distributions: Binomial, Poisson, geometric, negative binomial, hypergeometric

**Statistical Estimation**
- Confidence intervals for mean, proportion, variance

**Hypothesis Tests**
- Parametric tests: z-test, t-test, F-test, chi-squared test
- Nonparametric tests: Wilcoxon test, Mann-Whitney U test, Kruskal-Wallis test, Friedman test

**Effect Size**
- Cohen's d, Hedges' g, Glass's Δ
- Eta squared, omega squared
- Cohen's f, R²

**Resampling**
- Bootstrap
- Jackknife
- Permutation test

**Power Analysis**
- Sample size calculation and power for t-test
- Sample size calculation and power for proportion test

**Regression Analysis**
- Simple and multiple regression analysis
- Generalized linear models (GLM)

**ANOVA**
- One-way ANOVA
- Two-way ANOVA
- Repeated measures ANOVA

**Multivariate Analysis & Clustering**
- Basic multivariate analysis functions
- k-means clustering
- Hierarchical clustering

**Distance & Similarity**
- Euclidean distance, Manhattan distance, Chebyshev distance
- Minkowski distance
- Cosine similarity, Jaccard coefficient
- Hamming distance

**Numerical Utilities**
- Approximate equality for floating-point numbers
- Kahan summation algorithm
- Numerically stable logarithm and exponential functions
- Value clamping and range checking

**Additional Modules**
- Time series analysis
- Categorical data analysis
- Survival analysis
- Robust statistics
- Data wrangling (including missing value handling)

#### Documentation

- Complete API documentation via Doxygen
- Usage guide
- Sample programs for each module (30+)
- Installation guide
- Building and testing guide
- Contributing guide

#### Tests

- Comprehensive test suite with Google Test
- Unit tests for each module
- Edge case and exception tests

#### Other

- C++17 header-only library
- CMake support
- Cross-platform support (macOS, Linux, Windows)
- MIT License

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

[Unreleased]: https://github.com/yourusername/statcpp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/statcpp/releases/tag/v0.1.0
