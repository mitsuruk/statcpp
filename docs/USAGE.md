# Basic Usage

This document explains the basic usage patterns of the statcpp library.

## Quick Start

```cpp
#include <iostream>
#include <vector>
#include "statcpp/basic_statistics.hpp"

int main() {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};

    // Calculate mean
    double avg = statcpp::mean(data.begin(), data.end());
    std::cout << "Mean: " << avg << std::endl;  // 3.0

    return 0;
}
```

### Include Path Setup

statcpp is a header-only library. Headers are located under the `include/statcpp/` directory (or `include-ja/statcpp/` for Japanese-commented headers). Add the `include/` directory to your compiler's include path using the `-I` flag:

```bash
# Compile with the include path
g++ -std=c++17 -I/path/to/statcpp/include your_program.cpp -o your_program
```

Then include headers with the `statcpp/` prefix:

```cpp
#include "statcpp/basic_statistics.hpp"
```

When using CMake with `find_package(statcpp)` or `add_subdirectory`, the include path is configured automatically.

## Common Specifications

### Iterator Interface

All functions accept ranges via an iterator pair `(first, last)` representing the half-open interval $[first,\, last)$.
They work with any sequence that provides RandomAccessIterator, including `std::vector`, `std::array`, built-in arrays (`T a[N]`), and raw pointer ranges (`T* first, T* last`).

**Iterator Category**: All functions require **RandomAccessIterator**.
This is because they assume random access via `*(first + i)` and $O(1)$ computation of `std::distance`.
Non-RandomAccessIterators such as `std::forward_list` or input stream iterators are not supported.

```cpp
#include "statcpp/basic_statistics.hpp"
#include <vector>
#include <array>

// Using std::vector
std::vector<double> vec = {1.0, 2.0, 3.0};
double m1 = statcpp::mean(vec.begin(), vec.end());

// Using std::array
std::array<int, 5> arr = {1, 2, 3, 4, 5};
double m2 = statcpp::mean(arr.begin(), arr.end());

// Using C-style array
double data[] = {1.0, 2.0, 3.0};
double m3 = statcpp::mean(data, data + 3);
```

### Projection

Many functions have overloads that accept a projection function (callable such as a lambda) as an additional argument.
When a projection function $f$ is provided, the function evaluates $f(x_i)$ for each element $x_i$ and computes statistics on the results.

Supported functions:
- `sum`, `mean`, `median`, `mode`, `geometric_mean`, `harmonic_mean`, `trimmed_mean`
- `range`, `population_variance`, `sample_variance`, `variance`
- `population_stddev`, `sample_stddev`, `stddev`
- `coefficient_of_variation`, `iqr`, `mean_absolute_deviation`
- `minimum`, `maximum`, `quartiles`, `percentile`, `five_number_summary`

```cpp
#include "statcpp/basic_statistics.hpp"
#include <vector>

struct Product {
    std::string name;
    double price;
};

std::vector<Product> products = {
    {"Apple", 120.0},
    {"Banana", 80.0},
    {"Orange", 100.0}
};

// Calculate mean of prices (using projection)
double avg_price = statcpp::mean(
    products.begin(),
    products.end(),
    [](const Product& p) { return p.price; }
);
// avg_price = 100.0
```

### Pre-computed Mean

Functions that compute variance or standard deviation have overloads that accept a pre-computed mean value as `double precomputed_mean`.
This is useful for efficiently computing multiple statistics.

Supported functions:
- `population_variance`, `sample_variance`, `variance`
- `population_stddev`, `sample_stddev`, `stddev`
- `coefficient_of_variation`, `mean_absolute_deviation`

For projection versions, the argument order is `(first, last, proj, precomputed_mean)`.

```cpp
#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"
#include <vector>

std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};

// Compute mean only once
double avg = statcpp::mean(data.begin(), data.end());

// Use pre-computed mean to calculate variance and standard deviation
double var = statcpp::variance(data.begin(), data.end(), avg);
double sd = statcpp::stddev(data.begin(), data.end(), avg);
```

### Sorted Range

`median`, `trimmed_mean`, `iqr`, `quartiles`, `percentile`, and `five_number_summary` require a sorted range.
Pass a sequence (or iterator range) that has been sorted in ascending order beforehand.

The sort order is based on comparisons satisfying **strict weak ordering**, same as `std::sort`.

```cpp
#include "statcpp/basic_statistics.hpp"
#include "statcpp/order_statistics.hpp"
#include <vector>
#include <algorithm>

std::vector<double> data = {5.0, 2.0, 8.0, 1.0, 3.0};

// Sorting required
std::sort(data.begin(), data.end());

// Compute median on sorted data
double med = statcpp::median(data.begin(), data.end());
```

For projection versions, the elements must be arranged so that the **return values of projection function $f$** are in ascending order.

```cpp
struct Product {
    std::string name;
    double price;
};

std::vector<Product> products = {
    {"Apple", 120.0},
    {"Banana", 80.0},
    {"Orange", 100.0}
};

// Sort by projection key price before passing
std::sort(products.begin(), products.end(),
          [](const Product& a, const Product& b) { return a.price < b.price; });

auto q = statcpp::quartiles(products.begin(), products.end(),
                            [](const Product& p) { return p.price; });
```

### Quantile Linear Interpolation Method

Quantile computation in `iqr`, `quartiles`, `percentile`, and `five_number_summary` is based on linear interpolation (equivalent to R `type=7` / Excel `QUARTILE.INC` / `PERCENTILE.INC`).

For parameter $p\ （0 \leq p \leq 1）$, using 0-based index:

$$
\text{index} = p \times (n - 1)
$$

$$
lo = \lfloor \text{index} \rfloor, \quad frac = \text{index} - lo
$$

$$
Q = x[lo] \times (1 - frac) + x[lo + 1] \times frac
$$

**Endpoint handling**: When $lo + 1 \geq n$ (including $p = 1$ i.e., $lo = n - 1$), returns $Q = x[lo]$.
This ensures $p = 0$ returns the minimum and $p = 1$ returns the maximum.

### Exception Handling

For invalid arguments (out-of-range parameters, non-computable conditions, etc.) or when the definition of the statistic does not allow an empty range, `std::invalid_argument` is thrown.
The message includes the namespace-qualified function name (e.g., `"statcpp::mean: empty range"`).

However, `sum` and `count` allow empty ranges and return `value_type{}` and `0` respectively.

```cpp
#include "statcpp/basic_statistics.hpp"
#include <vector>
#include <stdexcept>

std::vector<double> empty_data;

try {
    double avg = statcpp::mean(empty_data.begin(), empty_data.end());
} catch (const std::invalid_argument& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    // Output: Error: statcpp::mean: empty range
}

// sum and count do not throw exceptions
double s = statcpp::sum(empty_data.begin(), empty_data.end());  // 0.0
std::size_t n = statcpp::count(empty_data.begin(), empty_data.end());  // 0
```

### Numerical Stability

Variance and standard deviation calculations are implemented using the two-pass algorithm (first compute the mean, then compute the sum of squared deviations).
Since Welford's online algorithm is not used, catastrophic cancellation may occur with data of vastly different scales.
Consider normalizing the data on the caller side when high precision is required.

```cpp
#include "statcpp/dispersion_spread.hpp"
#include <vector>

// Data with vastly different scales
std::vector<double> data = {1e10, 1e10 + 1, 1e10 + 2};

// Normalize before computing
double mean_val = statcpp::mean(data.begin(), data.end());
std::vector<double> normalized;
for (double x : data) {
    normalized.push_back(x - mean_val);
}

double var = statcpp::variance(normalized.begin(), normalized.end());
```

## Combining Multiple Modules

Example of using multiple headers together:

```cpp
#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"
#include "statcpp/order_statistics.hpp"
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<double> data = {5.0, 2.0, 8.0, 1.0, 3.0, 7.0, 4.0};

    // Basic statistics
    double avg = statcpp::mean(data.begin(), data.end());
    std::cout << "Mean: " << avg << std::endl;

    // Dispersion
    double sd = statcpp::stddev(data.begin(), data.end());
    std::cout << "Standard Deviation: " << sd << std::endl;

    // Order statistics (sorting required)
    std::sort(data.begin(), data.end());
    double med = statcpp::median(data.begin(), data.end());
    std::cout << "Median: " << med << std::endl;

    auto q = statcpp::quartiles(data.begin(), data.end());
    std::cout << "First Quartile: " << q.Q1 << std::endl;
    std::cout << "Second Quartile: " << q.Q2 << std::endl;
    std::cout << "Third Quartile: " << q.Q3 << std::endl;

    return 0;
}
```

## Next Steps

- For detailed function reference, see [API Reference](API_REFERENCE.md)
- For practical code examples, see [Examples](EXAMPLES.md)
- For building and testing, see [Building Guide](BUILDING.md)
