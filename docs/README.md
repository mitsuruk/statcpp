# statcpp — Statistics Library

C++17 Header-Only Statistics Library

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)

## Overview

statcpp is a header-only statistics library written in C++17. It provides a wide range of statistical functionality, from basic statistics to advanced hypothesis testing and regression analysis.

### Key Features

- **Header-Only**: No build required, just include and use
- **C++17 Standard Compliant**: Leverages modern C++ features
- **STL Style**: Intuitive iterator-based API
- **Projection Support**: Directly process struct members, etc.
- **Comprehensive Testing**: Extensive test suite with Google Test
- **Cross-Platform**: Tested on macOS and Linux

### Provided Functionality

- **Basic Statistics**: Mean, median, mode, variance, standard deviation, etc.
- **Order Statistics**: Quartiles, percentiles, five-number summary
- **Correlation Analysis**: Pearson, Spearman, Kendall correlation coefficients
- **Probability Distributions**: Normal, t, chi-squared, F, binomial, Poisson, etc.
- **Hypothesis Tests**: t-test, z-test, F-test, chi-squared test, Wilcoxon test, Mann-Whitney test, etc.
- **Effect Sizes**: Cohen's d, Hedges' g, eta squared, etc.
- **Regression Analysis**: Simple, multiple, logistic regression
- **ANOVA**: One-way, two-way, repeated measures ANOVA
- **Resampling**: Bootstrap, jackknife, permutation test
- **Power Analysis**: Sample size calculation and power analysis
- **Distance Metrics**: Euclidean distance, Manhattan distance, cosine similarity, etc.
- **Clustering**: k-means, hierarchical clustering

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/statcpp.git

# Add header files to include path
# Method 1: Install to system
cd statcpp
mkdir build && cd build
cmake ..
sudo cmake --install .

# Method 2: Copy to your project
cp -r statcpp/include/statcpp /your/project/include/
```

For details, see [Installation Guide](INSTALLATION.md).

### Basic Usage

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"
#include "statcpp/order_statistics.hpp"

int main() {
    std::vector<double> data = {5.0, 2.0, 8.0, 1.0, 3.0, 7.0, 4.0};

    // Basic statistics
    double avg = statcpp::mean(data.begin(), data.end());
    double sd = statcpp::stddev(data.begin(), data.end());

    std::cout << "Mean: " << avg << std::endl;              // 4.285...
    std::cout << "Standard Deviation: " << sd << std::endl; // 2.429...

    // Order statistics (sorting required)
    std::sort(data.begin(), data.end());
    double median = statcpp::median(data.begin(), data.end());
    auto q = statcpp::quartiles(data.begin(), data.end());

    std::cout << "Median: " << median << std::endl;         // 4.0
    std::cout << "First Quartile: " << q.Q1 << std::endl;   // 2.5
    std::cout << "Third Quartile: " << q.Q3 << std::endl;   // 6.5

    return 0;
}
```

Compile and run:

```bash
g++ -std=c++17 -I/path/to/statcpp/include example.cpp -o example
./example
```

### Using Projections

You can directly process struct members:

```cpp
#include "statcpp/basic_statistics.hpp"
#include <vector>

struct Product {
    std::string name;
    double price;
};

int main() {
    std::vector<Product> products = {
        {"Apple", 120.0},
        {"Banana", 80.0},
        {"Orange", 100.0}
    };

    // Calculate mean of prices
    double avg_price = statcpp::mean(
        products.begin(),
        products.end(),
        [](const Product& p) { return p.price; }
    );

    std::cout << "Average Price: " << avg_price << std::endl;  // 100.0
    return 0;
}
```

### Hypothesis Testing Example

```cpp
#include "statcpp/parametric_tests.hpp"
#include <vector>

int main() {
    std::vector<double> group1 = {23, 21, 19, 24, 20};
    std::vector<double> group2 = {31, 28, 30, 29, 32};

    // Two-sample t-test
    auto result = statcpp::t_test_two_sample(
        group1.begin(), group1.end(),
        group2.begin(), group2.end()
    );

    std::cout << "t-statistic: " << result.statistic << std::endl;
    std::cout << "p-value: " << result.p_value << std::endl;
    std::cout << "Degrees of Freedom: " << result.df << std::endl;

    if (result.p_value < 0.05) {
        std::cout << "Significant difference (p < 0.05)" << std::endl;
    }

    return 0;
}
```

## Module List

| Module | Header File | Description |
|--------|-------------|-------------|
| Basic Statistics | `basic_statistics.hpp` | Mean, median, mode, etc. |
| Dispersion | `dispersion_spread.hpp` | Variance, standard deviation, range, etc. |
| Order Statistics | `order_statistics.hpp` | Quartiles, percentiles, etc. |
| Distribution Shape | `shape_of_distribution.hpp` | Skewness, kurtosis |
| Correlation & Covariance | `correlation_covariance.hpp` | Correlation coefficients, covariance |
| Frequency Distribution | `frequency_distribution.hpp` | Histogram, frequency tables |
| Special Functions | `special_functions.hpp` | Gamma function, error function, etc. |
| Random Generation | `random_engine.hpp` | Random number generation engine |
| Continuous Distributions | `continuous_distributions.hpp` | Normal, t distribution, etc. |
| Discrete Distributions | `discrete_distributions.hpp` | Binomial, Poisson distribution, etc. |
| Estimation | `estimation.hpp` | Confidence interval calculation |
| Parametric Tests | `parametric_tests.hpp` | t-test, z-test, etc. |
| Nonparametric Tests | `nonparametric_tests.hpp` | Wilcoxon test, etc. |
| Effect Size | `effect_size.hpp` | Cohen's d, etc. |
| Resampling | `resampling.hpp` | Bootstrap, etc. |
| Power Analysis | `power_analysis.hpp` | Sample size calculation |
| Linear Regression | `linear_regression.hpp` | Simple, multiple regression |
| ANOVA | `anova.hpp` | Analysis of variance |
| GLM | `glm.hpp` | Logistic regression, etc. |
| Distance Metrics | `distance_metrics.hpp` | Euclidean distance, etc. |
| Numerical Utilities | `numerical_utils.hpp` | Numerical computation helpers |

Additional modules include: multivariate analysis, time series analysis, clustering, survival analysis, and more.

For details, see [API Reference](API_REFERENCE.md).

## Documentation

### Guides

- **[Installation](INSTALLATION.md)** - Installation methods and environment setup
- **[Usage](USAGE.md)** - Basic usage and common specifications
- **[Examples](EXAMPLES.md)** - Practical code examples
- **[API Reference](API_REFERENCE.md)** - Overview of all modules and functions
- **[Building and Testing](BUILDING.md)** - How to build tests and examples
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute to the project
- **[Changelog](CHANGELOG.md)** - Version history

### API Documentation

Detailed API documentation can be generated with Doxygen:

```bash
# Install Doxygen
brew install doxygen  # macOS
sudo apt-get install doxygen  # Ubuntu/Debian

# Generate documentation
./generate_docs.sh

# Open in browser
open doc/html/index.html  # macOS
xdg-open doc/html/index.html  # Linux
```

## Tested Environments

- macOS + Apple Clang 17.0.0
- macOS + GCC 15 (Homebrew)
- Ubuntu 24.04 ARM64 + GCC 13.3.0

## Development Purpose

During C++ development work, I often needed to perform statistical calculations and had accumulated a fair amount of code. I gathered these scattered pieces of code and created a simple statistics library as a header-only collection of functions.

The target programming language is C++17 with OS-independent code in mind.
A Rust version is also planned.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions to the project are welcome. Bug reports, feature requests, and pull requests are all appreciated.

For details, see [Contributing Guide](CONTRIBUTING.md).

## Support

- **Issue**: Bug reports and feature requests via [GitHub Issues](https://github.com/yourusername/statcpp/issues)
- **Discussion**: Questions and discussions via [GitHub Discussions](https://github.com/yourusername/statcpp/discussions)

## Acknowledgments

The following tools and AI were used in developing this project:

- **OpenAI ChatGPT 5.2** - Syntax checking and completeness review of documentation
- **Claude Code for VS Code Opus 4.5** - Google Test code generation, sample code fixes, refactoring
- **LM Studio google/gemma-2-27b** - Syntax checking and completeness review of documentation
- **llama.cpp** - Integrated build and error log management

---

**Note**: This library does not match commercial statistical software in terms of numerical stability and handling of extreme edge cases. When using for research or production environments, we recommend verifying results with other tools.
