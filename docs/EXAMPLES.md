# Examples

Sample programs demonstrating each feature of the statcpp library are provided in the `Examples/` directory.

## Sample Program List

### Basic Statistics & Dispersion

| Filename | Description |
|----------|-------------|
| `example_basic_statistics.cpp` | Basic statistics: mean, median, mode, etc. |
| `example_dispersion_spread.cpp` | Dispersion: variance, standard deviation, range, etc. |
| `example_order_statistics.cpp` | Quartiles, percentiles, five-number summary |
| `example_shape_of_distribution.cpp` | Distribution shape: skewness, kurtosis |
| `example_frequency_distribution.cpp` | Frequency distribution, histogram |

### Correlation & Covariance

| Filename | Description |
|----------|-------------|
| `example_correlation_covariance.cpp` | Correlation coefficients, covariance computation |

### Special Functions

| Filename | Description |
|----------|-------------|
| `example_special_functions.cpp` | Gamma function, beta function, error function, etc. |
| `example_numerical_utils.cpp` | Numerical utilities (Kahan summation, etc.) |

### Probability Distributions

| Filename | Description |
|----------|-------------|
| `example_random_engine.cpp` | Random number generation engine |
| `example_continuous_distributions.cpp` | Continuous distributions: normal, t, F, etc. |
| `example_discrete_distributions.cpp` | Discrete distributions: binomial, Poisson, etc. |

### Statistical Estimation & Tests

| Filename | Description |
|----------|-------------|
| `example_estimation.cpp` | Confidence interval estimation |
| `example_parametric_tests.cpp` | Parametric tests: t-test, z-test, etc. |
| `example_nonparametric_tests.cpp` | Nonparametric tests: Wilcoxon, Mann-Whitney, etc. |
| `example_effect_size.cpp` | Effect sizes: Cohen's d, correlation ratio, etc. |
| `example_power_analysis.cpp` | Power analysis, sample size calculation |

### Resampling & Regression Analysis

| Filename | Description |
|----------|-------------|
| `example_resampling.cpp` | Bootstrap, jackknife, permutation test |
| `example_linear_regression.cpp` | Simple and multiple regression analysis |
| `example_anova.cpp` | Analysis of variance (ANOVA) |
| `example_glm.cpp` | Generalized linear models (GLM) |

### Multivariate Analysis & Clustering

| Filename | Description |
|----------|-------------|
| `example_multivariate.cpp` | Multivariate analysis |
| `example_clustering.cpp` | k-means, hierarchical clustering |
| `example_distance_metrics.cpp` | Distance metrics: Euclidean, Manhattan, etc. |

### Time Series & Categorical Data

| Filename | Description |
|----------|-------------|
| `example_time_series.cpp` | Time series analysis |
| `example_categorical.cpp` | Categorical data analysis |

### Data Processing & Robust Statistics

| Filename | Description |
|----------|-------------|
| `example_data_wrangling.cpp` | Data transformation, missing value handling |
| `example_missing_data.cpp` | Advanced missing data handling |
| `example_robust.cpp` | Robust statistics (outlier-resistant statistics) |
| `example_survival.cpp` | Survival analysis |

### Other

| Filename | Description |
|----------|-------------|
| `example_model_selection.cpp` | Model selection (AIC, BIC, etc.) |

## Building and Running Examples

### Method 1: Using Shell Script (macOS/Linux)

Build and run all sample programs at once:

```bash
cd Examples
./build.sh
```

This script will:
1. Compile all `.cpp` files
2. Generate executables
3. Run each program and display output

### Method 2: Compile Individually

Build only a specific sample program:

```bash
cd Examples

# Using GCC
g++ -std=c++17 -I../include example_basic_statistics.cpp -o example_basic_statistics

# Using Clang
clang++ -std=c++17 -I../include example_basic_statistics.cpp -o example_basic_statistics

# Run
./example_basic_statistics
```

### Method 3: Using CMake

Build all examples with CMake:

```bash
cd Examples
mkdir build
cd build
cmake ..
cmake --build .

# Run
./example_basic_statistics
./example_dispersion_spread
# ...
```

## Understanding the Sample Code

Each sample program is structured as follows:

1. **Header Inclusion**: Include header files for the features being used
2. **Data Preparation**: Define sample data
3. **Function Calls**: Perform calculations using statcpp functions
4. **Result Display**: Output the calculation results

### Basic Example: `example_basic_statistics.cpp`

```cpp
#include <iostream>
#include <vector>
#include "statcpp/basic_statistics.hpp"

int main() {
    // 1. Data preparation
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};

    // 2. Function calls
    double avg = statcpp::mean(data.begin(), data.end());
    double total = statcpp::sum(data.begin(), data.end());
    std::size_t n = statcpp::count(data.begin(), data.end());

    // 3. Result display
    std::cout << "Count: " << n << std::endl;
    std::cout << "Sum: " << total << std::endl;
    std::cout << "Mean: " << avg << std::endl;

    return 0;
}
```

## Customization

The sample programs are designed for learning purposes. You can customize them for your own data and use cases as follows:

### Changing Data

```cpp
// Replace sample data with your own data
std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
// ↓
std::vector<double> data = {/* your data */};
```

### Reading Data from File

```cpp
#include <fstream>
#include <vector>

std::vector<double> read_data(const std::string& filename) {
    std::vector<double> data;
    std::ifstream file(filename);
    double value;
    while (file >> value) {
        data.push_back(value);
    }
    return data;
}

int main() {
    auto data = read_data("data.txt");
    double avg = statcpp::mean(data.begin(), data.end());
    // ...
}
```

### Computing Multiple Statistics at Once

```cpp
#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"
#include "statcpp/order_statistics.hpp"

void analyze_data(const std::vector<double>& data) {
    double avg = statcpp::mean(data.begin(), data.end());
    double sd = statcpp::stddev(data.begin(), data.end());

    auto sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    double med = statcpp::median(sorted_data.begin(), sorted_data.end());

    std::cout << "Mean: " << avg << std::endl;
    std::cout << "Standard Deviation: " << sd << std::endl;
    std::cout << "Median: " << med << std::endl;
}
```

## Troubleshooting

### Compile Error: Header Not Found

Verify that the include path is correctly specified:

```bash
g++ -std=c++17 -I/path/to/statcpp/include example.cpp -o example
```

### Runtime Error: invalid_argument

Many statistical functions throw exceptions for empty data or invalid arguments. Verify that your data is appropriate:

```cpp
std::vector<double> data = {/* data */};
if (data.empty()) {
    std::cerr << "Data is empty" << std::endl;
    return 1;
}

double avg = statcpp::mean(data.begin(), data.end());
```

### Forgetting to Sort

Functions that compute median or quartiles require sorted data:

```cpp
std::vector<double> data = {5.0, 2.0, 8.0, 1.0, 3.0};

// Sorting required
std::sort(data.begin(), data.end());

double med = statcpp::median(data.begin(), data.end());
```

## Next Steps

- For detailed specifications of each function, see [API Reference](API_REFERENCE.md)
- For basic usage, see [Usage Guide](USAGE.md)
- For test execution instructions, see [Building Guide](BUILDING.md)
