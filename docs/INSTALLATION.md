# Installation

statcpp is a C++17 header-only library. Simply add the header files to your include path and you're ready to use it.

Both English and Japanese commented versions are available. Choose the version that suits your preference.

## Requirements

- C++17 compatible compiler
  - GCC 7.0 or later
  - Clang 5.0 or later
  - Apple Clang 9.0 or later
- CMake 3.14 or later (optional, for building tests and examples)

## Tested Environments

- macOS + Apple Clang 17.0.0
- Ubuntu 24.04 ARM64 + GCC

## Installation Methods

### Method 1: Direct Use as Header-Only

The simplest method. Copy the `include/` or `include-ja/` directory to your project.

```bash
# Clone statcpp to any location
git clone https://github.com/yourusername/statcpp.git

# Copy the include directory to your project
# English version (default)
cp -r statcpp/include /path/to/your/project/

# Japanese version
cp -r statcpp/include-ja /path/to/your/project/include
```

Specify the include path when compiling:

```bash
g++ -std=c++17 -I/path/to/statcpp/include your_program.cpp -o your_program
```

### Method 2: Install with CMake

Install system-wide or to a specific location.

```bash
# Clone the repository
git clone https://github.com/yourusername/statcpp.git
cd statcpp

# Create build directory and install (English version - default)
mkdir build
cd build
cmake ..
sudo cmake --install .

# To install Japanese version instead
cmake .. -DSTATCPP_USE_JAPANESE=ON
sudo cmake --install .
```

By default, it installs to `/usr/local/include/`.

To specify a custom installation location:

```bash
cmake -DCMAKE_INSTALL_PREFIX=/your/custom/path ..
cmake --install .
```

#### CMake Options

| Option                       | Default | Description                                             |
| ---------------------------- | ------- | ------------------------------------------------------- |
| `STATCPP_USE_JAPANESE`       | OFF     | Use Japanese commented headers                          |
| `STATCPP_BUILD_TESTS`        | ON      | Build test suite                                        |
| `STATCPP_ENABLE_SANITIZERS`  | OFF     | Enable AddressSanitizer and UndefinedBehaviorSanitizer  |

### Method 3: Use as CMake Subdirectory

Add as a subdirectory to an existing CMake project.

```cmake
# CMakeLists.txt
add_subdirectory(external/statcpp)
target_link_libraries(your_target PRIVATE statcpp)
```

### Method 4: Use CMake FetchContent

With CMake 3.14 or later, you can use FetchContent to automatically download and use the library.

```cmake
include(FetchContent)

FetchContent_Declare(
    statcpp
    GIT_REPOSITORY https://github.com/yourusername/statcpp.git
    GIT_TAG        main  # or a specific tag/commit
)
FetchContent_MakeAvailable(statcpp)

target_link_libraries(your_target PRIVATE statcpp)
```

## Using After Installation

After installation, headers are placed in `/usr/local/include/statcpp/` (or your custom prefix).

Include headers with the `statcpp/` prefix:

```bash
g++ -std=c++17 your_program.cpp -o your_program
```

```cpp
#include "statcpp/basic_statistics.hpp"
```

## Verifying Installation

You can verify the installation with this simple program.

```cpp
#include <iostream>
#include <vector>
#include "statcpp/basic_statistics.hpp"

int main() {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    double avg = statcpp::mean(data.begin(), data.end());
    std::cout << "Mean: " << avg << std::endl;  // Output: Mean: 3
    return 0;
}
```

Compile and run:

```bash
g++ -std=c++17 -I/path/to/statcpp/include test.cpp -o test
./test
```

## Troubleshooting

### Compiler Does Not Support C++17

Please upgrade to a newer version of the compiler.

```bash
# macOS (Homebrew)
brew install gcc

# Ubuntu/Debian
sudo apt-get install g++-9

# Compile with specific version
g++-9 -std=c++17 your_program.cpp -o your_program
```

### Include Path Not Found

Explicitly specify the path using the `-I` option when compiling.

```bash
g++ -std=c++17 -I/usr/local/include your_program.cpp -o your_program
```

### CMake Cannot Find the Library

Set `CMAKE_PREFIX_PATH`.

```bash
cmake -DCMAKE_PREFIX_PATH=/your/custom/install/path ..
```
