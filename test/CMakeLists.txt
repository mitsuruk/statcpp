# Use system GoogleTest
find_package(GTest REQUIRED)

# Create test executable
add_executable(statcpp_tests
    # Module 1: Descriptive Statistics
    test_basic_statistics.cpp
    test_dispersion_spread.cpp
    test_order_statistics.cpp
    test_shape_of_distribution.cpp
    test_correlation_covariance.cpp
    test_frequency_distribution.cpp
    # Module 2: Probability Distributions
    test_special_functions.cpp
    test_random_engine.cpp
    test_continuous_distributions.cpp
    test_discrete_distributions.cpp
    # Module 3: Inferential Statistics
    test_estimation.cpp
    test_parametric_tests.cpp
    test_nonparametric_tests.cpp
    test_effect_size.cpp
    test_resampling.cpp
    test_power_analysis.cpp
    # Module 4: Statistical Modeling
    test_linear_regression.cpp
    test_anova.cpp
    test_glm.cpp
    test_model_selection.cpp
    # Module 5: Applied Analysis
    test_multivariate.cpp
    test_time_series.cpp
    test_categorical.cpp
    test_survival.cpp
    test_robust.cpp
    test_clustering.cpp
    # Module 6: Data Infrastructure
    test_data_wrangling.cpp
    test_missing_data.cpp
)

target_link_libraries(statcpp_tests PRIVATE
    statcpp
    GTest::gtest_main
)

# Additional settings when sanitizers are enabled
if(STATCPP_ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(statcpp_tests PRIVATE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
            -g
        )
        target_link_options(statcpp_tests PRIVATE
            -fsanitize=address,undefined
        )
    endif()
endif()

# Integration with CTest
include(GoogleTest)
gtest_discover_tests(statcpp_tests)

# Select example directory based on language option
if(STATCPP_USE_JAPANESE)
    set(EXAMPLE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../examples-ja)
else()
    set(EXAMPLE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../examples)
endif()

# Build example programs
set(EXAMPLE_SOURCES
    ${EXAMPLE_DIR}/example_basic_statistics.cpp
    ${EXAMPLE_DIR}/example_dispersion_spread.cpp
    ${EXAMPLE_DIR}/example_order_statistics.cpp
    ${EXAMPLE_DIR}/example_shape_of_distribution.cpp
    ${EXAMPLE_DIR}/example_correlation_covariance.cpp
    ${EXAMPLE_DIR}/example_frequency_distribution.cpp
    ${EXAMPLE_DIR}/example_continuous_distributions.cpp
    ${EXAMPLE_DIR}/example_discrete_distributions.cpp
    ${EXAMPLE_DIR}/example_estimation.cpp
    ${EXAMPLE_DIR}/example_parametric_tests.cpp
    ${EXAMPLE_DIR}/example_nonparametric_tests.cpp
    ${EXAMPLE_DIR}/example_resampling.cpp
    ${EXAMPLE_DIR}/example_linear_regression.cpp
    ${EXAMPLE_DIR}/example_anova.cpp
    ${EXAMPLE_DIR}/example_distance_metrics.cpp
    ${EXAMPLE_DIR}/example_numerical_utils.cpp
    ${EXAMPLE_DIR}/example_power_analysis.cpp
    ${EXAMPLE_DIR}/example_missing_data.cpp
    ${EXAMPLE_DIR}/example_data_wrangling.cpp
    ${EXAMPLE_DIR}/example_special_functions.cpp
    ${EXAMPLE_DIR}/example_random_engine.cpp
    ${EXAMPLE_DIR}/example_effect_size.cpp
    ${EXAMPLE_DIR}/example_categorical.cpp
    ${EXAMPLE_DIR}/example_robust.cpp
    ${EXAMPLE_DIR}/example_clustering.cpp
    ${EXAMPLE_DIR}/example_glm.cpp
    ${EXAMPLE_DIR}/example_multivariate.cpp
    ${EXAMPLE_DIR}/example_model_selection.cpp
    ${EXAMPLE_DIR}/example_survival.cpp
    ${EXAMPLE_DIR}/example_time_series.cpp
)

foreach(example_source ${EXAMPLE_SOURCES})
    get_filename_component(example_name ${example_source} NAME_WE)
    add_executable(${example_name} ${example_source})
    target_link_libraries(${example_name} PRIVATE statcpp)

    # Additional settings when sanitizers are enabled
    if(STATCPP_ENABLE_SANITIZERS)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
            target_compile_options(${example_name} PRIVATE
                -fsanitize=address,undefined
                -fno-omit-frame-pointer
                -g
            )
            target_link_options(${example_name} PRIVATE
                -fsanitize=address,undefined
            )
        endif()
    endif()
endforeach()
