# Building and Testing

Since statcpp is a header-only library, there is no need to build the library itself.
This document explains how to build and run tests, build example programs, and generate documentation.

## Requirements

### Basic Requirements

- C++17 compatible compiler
  - GCC 7.0 or later
  - Clang 5.0 or later
  - Apple Clang 9.0 or later

### For Building Tests

- CMake 3.14 or later
- Google Test (must be pre-installed on the system)

### For Generating Documentation

- [Doxygen](https://www.doxygen.nl/)

## Building and Running Tests

### Method 1: Using CMake (Recommended)

```bash
# Run in the project root directory
mkdir build
cd build

# CMake configuration (enable tests)
cmake -DSTATCPP_BUILD_TESTS=ON ..

# Build
cmake --build .

# Run tests
ctest --output-on-failure
```

Or:

```bash
# Run directly from the test directory
cd test
mkdir build
cd build
cmake ..
cmake --build .
ctest --output-on-failure
```

### Method 2: Run Specific Tests Only

```bash
cd build

# Run specific test files
./test/test_basic_statistics
./test/test_dispersion_spread
./test/test_order_statistics
# ...
```

### Available Tests

The project includes the following test suites:

- `test_basic_statistics` - Basic statistics
- `test_dispersion_spread` - Dispersion
- `test_order_statistics` - Order statistics
- `test_shape_of_distribution` - Distribution shape
- `test_correlation_covariance` - Correlation & covariance
- `test_frequency_distribution` - Frequency distribution
- `test_special_functions` - Special functions
- `test_random_engine` - Random number generation
- `test_continuous_distributions` - Continuous distributions
- `test_discrete_distributions` - Discrete distributions
- `test_estimation` - Estimation
- `test_parametric_tests` - Parametric tests
- `test_nonparametric_tests` - Nonparametric tests
- `test_effect_size` - Effect size
- `test_resampling` - Resampling
- `test_power_analysis` - Power analysis
- `test_linear_regression` - Linear regression
- `test_anova` - ANOVA
- `test_glm` - Generalized linear models
- `test_model_selection` - Model selection
- `test_multivariate` - Multivariate analysis
- `test_time_series` - Time series analysis
- `test_categorical` - Categorical data analysis
- `test_survival` - Survival analysis
- `test_robust` - Robust statistics
- `test_clustering` - Clustering
- `test_data_wrangling` - Data transformation
- `test_missing_data` - Missing data

## Building Example Programs

### Method 1: Using Shell Script (macOS/Linux)

The simplest method:

```bash
cd Examples
./build.sh
```

This script will:
1. Compile all `.cpp` files
2. Generate executables
3. Run each program and display output

### Method 2: Using CMake

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

### Method 3: Compile Individually

```bash
cd Examples

# GCC
g++ -std=c++17 -I../include example_basic_statistics.cpp -o example_basic_statistics

# Clang
clang++ -std=c++17 -I../include example_basic_statistics.cpp -o example_basic_statistics

# Apple Clang (macOS)
clang++ -std=c++17 -I../include example_basic_statistics.cpp -o example_basic_statistics

# Run
./example_basic_statistics
```

## Generating Documentation

### Installing Doxygen

```bash
# macOS (Homebrew)
brew install doxygen

# Ubuntu/Debian
sudo apt-get install doxygen

# Windows
# Download from https://www.doxygen.nl/download.html
```

### Generating Documentation

```bash
# Run in the project root directory

# Using shell script (recommended)
./generate_docs.sh

# Or run Doxygen directly
doxygen Doxyfile
```

The generated documentation is output to `doc/html/index.html`.

### Viewing Documentation

```bash
# macOS
open doc/html/index.html

# Linux
xdg-open doc/html/index.html

# Windows
start doc/html/index.html
```

## CMake Build Options

### Disabling Tests

```bash
cmake -DSTATCPP_BUILD_TESTS=OFF ..
```

### Enabling Sanitizers

Enable sanitizers for detecting memory errors and undefined behavior:

```bash
cmake -DSTATCPP_ENABLE_SANITIZERS=ON ..
cmake --build .
ctest --output-on-failure
```

**Note**: Use sanitizers only during development and debugging. Disable them for production environments.

### Specifying a Custom Compiler

```bash
# Specify GCC
cmake -DCMAKE_CXX_COMPILER=g++-11 ..

# Specify Clang
cmake -DCMAKE_CXX_COMPILER=clang++-14 ..
```

### Specifying Build Type

```bash
# Debug build (with debug symbols)
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release build (optimizations enabled)
cmake -DCMAKE_BUILD_TYPE=Release ..

# RelWithDebInfo (optimizations + debug symbols)
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
```

## Troubleshooting

### CMake Not Found

```bash
# macOS
brew install cmake

# Ubuntu/Debian
sudo apt-get install cmake

# Or download from the official site
# https://cmake.org/download/
```

### Google Test Not Found

Google Test must be pre-installed on the system.

```bash
# Ubuntu/Debian
sudo apt-get install libgtest-dev

# macOS
brew install googletest

# Windows (vcpkg)
vcpkg install gtest
```

### Compiler Error: No C++17 Support

You may be using an older compiler. Please upgrade to the latest version:

```bash
# Ubuntu/Debian
sudo apt-get install g++-9

# macOS
brew install gcc
```

### Tests Fail

1. **Numerical precision issues**: Tests may fail on some environments due to floating-point rounding errors.
2. **Random seeds**: Tests using random numbers may produce different results with different seeds.

View detailed error output:

```bash
ctest --output-on-failure --verbose
```

Run specific tests directly:

```bash
./test/test_basic_statistics --gtest_filter=MeanTest.*
```

### Warnings During Documentation Generation

Doxygen warnings can usually be ignored. If errors occur:

1. Check Doxygen version (1.9.0 or later recommended)
2. Check the `Doxyfile` configuration

## Continuous Integration (CI)

Reference for setting up CI for the project:

### GitHub Actions Example

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: sudo apt-get install -y cmake g++
      - name: Build and test
        run: |
          mkdir build
          cd build
          cmake -DSTATCPP_BUILD_TESTS=ON ..
          cmake --build .
          ctest --output-on-failure
```

## Tested Environments

This library has been tested on:

- **macOS + Apple Clang 17.0.0**
- **Ubuntu 24.04 ARM64 + GCC**

## Next Steps

- For details on sample code, see [Examples](EXAMPLES.md)
- For contribution guidelines, see [Contributing Guide](CONTRIBUTING.md)
- For API details, see [API Reference](API_REFERENCE.md)
