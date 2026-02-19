cmake_minimum_required(VERSION 3.14)
project(statcpp_examples
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "Examples for statcpp - C++17 Header-only Statistics Library"
)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Use g++-15 explicitly
set(CMAKE_CXX_COMPILER "/opt/homebrew/bin/g++-15")
set(CMAKE_C_COMPILER "/opt/homebrew/bin/gcc-15")

# Include the statcpp headers from parent directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Enable compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# List of all example source files
set(EXAMPLE_SOURCES
    example_basic_statistics.cpp
    example_dispersion_spread.cpp
    example_order_statistics.cpp
    example_shape_of_distribution.cpp
    example_correlation_covariance.cpp
    example_frequency_distribution.cpp
    example_continuous_distributions.cpp
    example_discrete_distributions.cpp
    example_estimation.cpp
    example_parametric_tests.cpp
    example_nonparametric_tests.cpp
    example_resampling.cpp
    example_linear_regression.cpp
    example_anova.cpp
    example_distance_metrics.cpp
    example_numerical_utils.cpp
    example_power_analysis.cpp
    example_missing_data.cpp
    example_data_wrangling.cpp
    example_special_functions.cpp
    example_random_engine.cpp
    example_effect_size.cpp
    example_categorical.cpp
    example_robust.cpp
    example_clustering.cpp
    example_glm.cpp
    example_multivariate.cpp
    example_model_selection.cpp
    example_survival.cpp
    example_time_series.cpp
)

# Create an executable for each example
foreach(example_source ${EXAMPLE_SOURCES})
    # Extract the executable name from the source file
    get_filename_component(example_name ${example_source} NAME_WE)

    # Create executable
    add_executable(${example_name} ${example_source})

    # No additional libraries needed (header-only library)
endforeach()

# Custom target to build all examples
add_custom_target(examples
    DEPENDS ${EXAMPLE_SOURCES}
    COMMENT "Building all example programs"
)

# Print build information
message(STATUS "statcpp Examples Configuration:")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  C++ Standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Include Directory: ${CMAKE_CURRENT_SOURCE_DIR}/../include")
message(STATUS "  Number of Examples: ${CMAKE_CURRENT_LIST_LENGTH}")
