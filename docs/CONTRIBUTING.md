# Contributing Guide

Contributions to the statcpp project are welcome. This document explains how to contribute and the guidelines to follow.

## Ways to Contribute

### 1. Bug Reports

If you find a bug, please create an Issue with the following information:

- **Environment Information**
  - OS and version
  - Compiler and version
  - C++ standard (C++17, C++20, etc.)
- **Steps to Reproduce**
  - Minimal reproducible code example
  - Expected behavior
  - Actual behavior
- **Error Messages**
  - Full text of compile or runtime errors

### 2. Feature Requests

Proposals for new features or improvements are very welcome:

- **Background and Motivation**
  - Why is this feature needed?
  - What use cases would benefit from it?
- **Proposal**
  - API design proposal
  - Code snippet showing usage
- **Alternatives**
  - Consideration of other implementation approaches

### 3. Pull Requests

To contribute code, follow these steps:

1. **Fork and Clone**
   ```bash
   # Fork the repository
   git clone https://github.com/yourusername/statcpp.git
   cd statcpp
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

3. **Make Changes**
   - Implement code
   - Add tests
   - Update documentation

4. **Run Tests**
   ```bash
   mkdir build
   cd build
   cmake -DSTATCPP_BUILD_TESTS=ON ..
   cmake --build .
   ctest --output-on-failure
   ```

5. **Commit**
   ```bash
   git add .
   git commit -m "Add: brief description of new feature"
   ```

6. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```
   Create a pull request on GitHub

## Coding Standards

### File Organization

- **Header files**: Place in `include/statcpp/`
- **Test files**: Place in `test/` (format: `test_*.cpp`)
- **Example files**: Place in `Examples/` (format: `example_*.cpp`)

### Coding Style

#### Naming Conventions

```cpp
// Namespace: lowercase
namespace statcpp {

// Function names: snake_case (lowercase + underscore)
double mean(Iterator first, Iterator last);
double standard_deviation(Iterator first, Iterator last);

// Struct/class names: PascalCase (first letter of each word capitalized)
struct TestResult {
    double statistic;
    double p_value;
};

// Variable names: snake_case
double sample_mean = 0.0;
std::size_t sample_size = 100;

// Constants: UPPER_CASE + underscore
constexpr double DEFAULT_ALPHA = 0.05;

// Template parameters: PascalCase
template <typename Iterator, typename Projection>
double mean(Iterator first, Iterator last, Projection proj);
}
```

#### Indentation and Formatting

```cpp
// Indentation: 4 spaces
void example_function() {
    if (condition) {
        // code
    }
}

// Brace position
void function() {  // same line
    // code
}

struct MyStruct {  // same line
    int value;
};
```

#### Header File Structure

```cpp
#ifndef STATCPP_MODULE_NAME_HPP
#define STATCPP_MODULE_NAME_HPP

#include <vector>
#include <algorithm>
// Other standard library headers

namespace statcpp {

/**
 * @brief Brief description of function
 *
 * Detailed description (include formulas and usage examples as needed)
 *
 * @tparam Iterator Random access iterator
 * @param first Iterator to the beginning of the range
 * @param last Iterator to the end of the range
 * @return Computed result
 * @throws std::invalid_argument If input is invalid
 */
template <typename Iterator>
double function_name(Iterator first, Iterator last) {
    // Implementation
}

}  // namespace statcpp

#endif  // STATCPP_MODULE_NAME_HPP
```

### Documentation

All public functions should have Doxygen-style comments:

```cpp
/**
 * @brief Compute arithmetic mean
 *
 * Computes the arithmetic mean of elements in the iterator range [first, last).
 *
 * Formula:
 * \f[
 * \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
 * \f]
 *
 * @tparam Iterator Random access iterator
 * @param first Beginning of range
 * @param last End of range (element not included)
 * @return Mean value (double type)
 * @throws std::invalid_argument If range is empty
 *
 * Usage example:
 * @code
 * std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
 * double avg = statcpp::mean(data.begin(), data.end());
 * // avg = 3.0
 * @endcode
 */
template <typename Iterator>
double mean(Iterator first, Iterator last);
```

### Writing Tests

Add tests for all new features:

```cpp
#include <gtest/gtest.h>
#include "statcpp/module_name.hpp"
#include <vector>

// Test case name: ModuleName + Test
// Test name: Description of functionality
TEST(ModuleNameTest, BasicFunctionality) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    double result = statcpp::function_name(data.begin(), data.end());
    EXPECT_DOUBLE_EQ(result, 3.0);
}

TEST(ModuleNameTest, EmptyRange) {
    std::vector<double> empty;
    EXPECT_THROW(
        statcpp::function_name(empty.begin(), empty.end()),
        std::invalid_argument
    );
}

TEST(ModuleNameTest, SingleElement) {
    std::vector<double> single = {42.0};
    double result = statcpp::function_name(single.begin(), single.end());
    EXPECT_DOUBLE_EQ(result, 42.0);
}

// Approximate comparison for floating-point numbers
TEST(ModuleNameTest, FloatingPointComparison) {
    std::vector<double> data = {1.1, 2.2, 3.3};
    double result = statcpp::function_name(data.begin(), data.end());
    EXPECT_NEAR(result, 2.2, 1e-10);  // tolerance 1e-10
}
```

### Error Handling

```cpp
// Throw std::invalid_argument for invalid input
template <typename Iterator>
double mean(Iterator first, Iterator last) {
    if (first == last) {
        throw std::invalid_argument("statcpp::mean: empty range");
    }
    // Implementation
}

// Error messages should include namespace-qualified function name
throw std::invalid_argument("statcpp::function_name: error description");
```

## Development Best Practices

### 1. Numerical Stability

Pay attention to numerical stability in floating-point operations:

```cpp
// Bad example: prone to catastrophic cancellation
double variance = (sum_of_squares / n) - (mean * mean);

// Good example: two-pass method
double mean_val = mean(first, last);
double sum_sq_dev = 0.0;
for (auto it = first; it != last; ++it) {
    double dev = *it - mean_val;
    sum_sq_dev += dev * dev;
}
double variance = sum_sq_dev / n;
```

### 2. Iterator Requirements

All functions require RandomAccessIterator:

```cpp
template <typename Iterator>
double function_name(Iterator first, Iterator last) {
    static_assert(
        std::is_base_of_v<
            std::random_access_iterator_tag,
            typename std::iterator_traits<Iterator>::iterator_category
        >,
        "RandomAccessIterator required"
    );
    // Implementation
}
```

### 3. Projection Support

Provide projection versions whenever possible:

```cpp
// Basic version
template <typename Iterator>
double mean(Iterator first, Iterator last);

// Projection version
template <typename Iterator, typename Projection>
double mean(Iterator first, Iterator last, Projection proj) {
    // Use proj to transform values
}
```

### 4. Performance

- Avoid unnecessary copies (use const references)
- Use `inline` when possible
- Implement templates entirely in headers

```cpp
template <typename Iterator>
inline double mean(Iterator first, Iterator last) {
    // Implementation
}
```

## Adding Example Code

Add usage examples for new features:

1. Create `example_your_feature.cpp` in `Examples/`
2. Include practical usage examples
3. Add explanatory comments

```cpp
#include <iostream>
#include <vector>
#include "statcpp/your_module.hpp"

int main() {
    std::cout << "=== Your Feature Example ===" << std::endl;

    // Sample data
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};

    // Usage example
    double result = statcpp::your_function(data.begin(), data.end());
    std::cout << "Result: " << result << std::endl;

    return 0;
}
```

## Updating Documentation

Update the following documentation as code changes:

- `docs/API_REFERENCE.md` - Add new functions
- `docs/EXAMPLES.md` - Add new sample programs
- `docs/CHANGELOG.md` - Record changes

## Commit Messages

Use clear and concise commit messages:

```
Add: New feature addition
Fix: Bug fix
Update: Improvement to existing feature
Refactor: Refactoring
Test: Add/modify tests
Docs: Documentation update
```

Examples:
```
Add: weighted_variance function to dispersion_spread module
Fix: numeric overflow in variance calculation for large datasets
Update: improve numerical stability of stddev function
Test: add edge case tests for empty ranges
Docs: update API reference with new functions
```

## Review Process

Pull requests are reviewed for the following:

1. **Code Quality**
   - Does it comply with coding standards?
   - Is there appropriate error handling?
2. **Tests**
   - Do all tests pass?
   - Are there tests for new features?
3. **Documentation**
   - Are there Doxygen comments?
   - Is documentation updated?
4. **Compatibility**
   - Are there no breaking changes to the existing API?

## Questions and Support

- **Issue**: Bug reports and feature requests via GitHub Issues
- **Discussion**: General questions and discussions via GitHub Discussions

## License

Contributions are subject to the project's license (MIT License).
By creating a pull request, you agree that your contribution will be distributed under this license.

## Acknowledgments

Thank you for contributing to the project. Your contributions make statcpp a better library.
