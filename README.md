# statcpp — Statistics Library for C++

C++17 Header-only Statistics Library

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)

[日本語版 / Japanese](README.ja.md)

## Overview

statcpp is a header-only statistics library written in C++17. It provides 524 public functions across 31 header files, covering a wide range of statistical functionality from basic statistics to advanced hypothesis testing and regression analysis. The library includes 758 unit tests and 167 R-verified numerical checks.

### Key Features

- **524 public functions**: Comprehensive coverage across 31 modules
- **Header-only**: No build required, just include and use
- **C++17 compliant**: Leverages modern C++ features
- **STL-style**: Intuitive iterator-based API
- **Projection support**: Directly process struct members and other data
- **Comprehensive testing**: 758 unit tests with Google Test, 167 numerical checks verified against R 4.4.2
- **Cross-platform**: Tested on macOS and Linux
- **Bilingual support**: English and Japanese commented headers available

### Provided Functions

- **Basic statistics**: Mean, median, mode, variance, standard deviation, etc.
- **Order statistics**: Quartiles, percentiles, five-number summary
- **Correlation analysis**: Pearson, Spearman, Kendall correlation coefficients
- **Probability distributions**: Normal, t, chi-square, F, binomial, Poisson, etc.
- **Hypothesis tests**: t-test, z-test, F-test, chi-square test, Wilcoxon test, Mann-Whitney test, etc.
- **Effect sizes**: Cohen's d, Hedges' g, eta-squared, etc.
- **Regression analysis**: Simple regression, multiple regression, logistic regression
- **Analysis of variance**: One-way, two-way, repeated measures ANOVA
- **Resampling**: Bootstrap, jackknife, permutation tests
- **Power analysis**: Sample size calculation and power analysis
- **Distance metrics**: Euclidean distance, Manhattan distance, cosine similarity, etc.
- **Clustering**: k-means, hierarchical clustering

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/mitsuruk/statcpp.git

# Add header files to include path
# Method 1: Install to system (English version - default)
cd statcpp
mkdir build && cd build
cmake ..
sudo cmake --install .

# Method 1b: Install Japanese version
cmake .. -DSTATCPP_USE_JAPANESE=ON
sudo cmake --install .

# Method 2: Copy to your project
cp -r statcpp/include /your/project/        # English
cp -r statcpp/include-ja /your/project/     # Japanese
```

See [Installation Guide](docs/INSTALLATION.md) for details.

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
    std::cout << "Standard deviation: " << sd << std::endl;  // 2.429...

    // Order statistics (requires sorting)
    std::sort(data.begin(), data.end());
    double median = statcpp::median(data.begin(), data.end());
    auto q = statcpp::quartiles(data.begin(), data.end());

    std::cout << "Median: " << median << std::endl;         // 4.0
    std::cout << "First quartile: " << q.Q1 << std::endl;   // 2.5
    std::cout << "Third quartile: " << q.Q3 << std::endl;   // 6.5

    return 0;
}
```

Compile and run:

```bash
g++ -std=c++17 -I/path/to/statcpp/include example.cpp -o example
./example
```

### Example with Projections

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

    // Calculate mean price
    double avg_price = statcpp::mean(
        products.begin(),
        products.end(),
        [](const Product& p) { return p.price; }
    );

    std::cout << "Average price: " << avg_price << std::endl;  // 100.0
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
    std::cout << "Degrees of freedom: " << result.df << std::endl;

    if (result.p_value < 0.05) {
        std::cout << "Significant difference (p < 0.05)" << std::endl;
    }

    return 0;
}
```

## Module List

| Module | Header File | Contents |
|--------|-------------|----------|
| Basic Statistics | `basic_statistics.hpp` | Mean, median, mode, etc. |
| Dispersion | `dispersion_spread.hpp` | Variance, standard deviation, range, etc. |
| Order Statistics | `order_statistics.hpp` | Quartiles, percentiles, etc. |
| Distribution Shape | `shape_of_distribution.hpp` | Skewness, kurtosis |
| Correlation & Covariance | `correlation_covariance.hpp` | Correlation coefficients, covariance |
| Frequency Distribution | `frequency_distribution.hpp` | Histogram, frequency tables |
| Special Functions | `special_functions.hpp` | Gamma function, error function, etc. |
| Random Number Generation | `random_engine.hpp` | Random number generation engine |
| Continuous Distributions | `continuous_distributions.hpp` | Normal distribution, t-distribution, etc. |
| Discrete Distributions | `discrete_distributions.hpp` | Binomial distribution, Poisson distribution, etc. |
| Estimation | `estimation.hpp` | Confidence interval calculation |
| Parametric Tests | `parametric_tests.hpp` | t-test, z-test, etc. |
| Nonparametric Tests | `nonparametric_tests.hpp` | Wilcoxon test, etc. |
| Effect Size | `effect_size.hpp` | Cohen's d, etc. |
| Resampling | `resampling.hpp` | Bootstrap, etc. |
| Power Analysis | `power_analysis.hpp` | Sample size calculation |
| Linear Regression | `linear_regression.hpp` | Simple regression, multiple regression |
| Analysis of Variance | `anova.hpp` | ANOVA |
| Generalized Linear Models | `glm.hpp` | Logistic regression, etc. |
| Distance Metrics | `distance_metrics.hpp` | Euclidean distance, etc. |
| Numerical Utilities | `numerical_utils.hpp` | Numerical calculation helpers |

Also includes: multivariate analysis, time series analysis, clustering, survival analysis, and more.

See [API Reference](docs/API_REFERENCE.md) for details.

## Documentation

### Guides (English)

- **[Installation](docs/INSTALLATION.md)** - Installation instructions and environment setup
- **[Usage](docs/USAGE.md)** - Basic usage and common specifications
- **[Examples](docs/EXAMPLES.md)** - Practical code examples
- **[API Reference](docs/API_REFERENCE.md)** - Overview of all modules and functions
- **[Building and Testing](docs/BUILDING.md)** - How to build tests and examples
- **[Contributing Guide](docs/CONTRIBUTING.md)** - How to contribute to the project
- **[Changelog](docs/CHANGELOG.md)** - Version history
- **[TODO](todo.md)** - Development plans and improvement items

### Guides (Japanese / 日本語ドキュメント)

- **[インストール](docs-ja/INSTALLATION.md)** - インストール手順と環境構築
- **[使い方](docs-ja/USAGE.md)** - 基本的な使い方と共通仕様
- **[サンプル集](docs-ja/EXAMPLES.md)** - 実践的なコード例
- **[APIリファレンス](docs-ja/API_REFERENCE.md)** - 全モジュール・関数の概要
- **[ビルドとテスト](docs-ja/BUILDING.md)** - テストとサンプルのビルド方法
- **[コントリビューションガイド](docs-ja/CONTRIBUTING.md)** - プロジェクトへの貢献方法
- **[変更履歴](docs-ja/CHANGELOG.md)** - バージョン履歴
- **[TODO](todo.ja.md)** - 開発計画と改善項目

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

I had accumulated code for statistical calculations during C++ development work. This project consolidates that scattered code into a simple header-only statistics library.

The target programming language is C++17 with OS-independent code. A Rust version is also planned.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions to the project are welcome. Feel free to submit bug reports, feature requests, and pull requests.

See the [Contributing Guide](docs/CONTRIBUTING.md) for details.

## Support

- **Issues**: Report bugs and feature requests on [GitHub Issues](https://github.com/mitsuruk/statcpp/issues)
- **Discussions**: Questions and discussions on [GitHub Discussions](https://github.com/mitsuruk/statcpp/discussions)

## Acknowledgments

This project was developed using the following tools and AI:

- **OpenAI ChatGPT 5.2** - Document syntax checking, identifying missing explanations
- **Claude Code for VS Code Opus 4.5** - Google Test code generation, sample code fixes, refactoring
- **LM Studio google/gemma-2-27b** - Document syntax checking, identifying missing explanations
- **llmama.cpp/gemma-2-27b** - Integrated build and error log management

---

**Note**: This library does not have the same level of numerical stability or edge case handling as commercial statistical software. When using for research or production, we recommend verifying results with other tools.
