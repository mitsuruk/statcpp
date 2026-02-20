/**
 * @file statcpp.hpp
 * @brief Main header file for the statcpp library
 *
 * An integrated header that includes all modules of the statistical analysis library.
 * Including this file provides access to all statistical functionality.
 */

#pragma once

// Module 1: Descriptive Statistics
#include "statcpp/basic_statistics.hpp"
#include "statcpp/order_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"
#include "statcpp/shape_of_distribution.hpp"
#include "statcpp/correlation_covariance.hpp"
#include "statcpp/frequency_distribution.hpp"

// Module 2: Probability Distributions
#include "statcpp/special_functions.hpp"
#include "statcpp/random_engine.hpp"
#include "statcpp/continuous_distributions.hpp"
#include "statcpp/discrete_distributions.hpp"

// Module 3: Inferential Statistics
#include "statcpp/estimation.hpp"
#include "statcpp/parametric_tests.hpp"
#include "statcpp/nonparametric_tests.hpp"
#include "statcpp/effect_size.hpp"
#include "statcpp/resampling.hpp"

// Module 4: Statistical Modeling
#include "statcpp/linear_regression.hpp"
#include "statcpp/anova.hpp"
#include "statcpp/glm.hpp"
#include "statcpp/model_selection.hpp"

// Module 5: Applied Analysis
#include "statcpp/distance_metrics.hpp"
#include "statcpp/multivariate.hpp"
#include "statcpp/clustering.hpp"
#include "statcpp/time_series.hpp"
#include "statcpp/categorical.hpp"
#include "statcpp/survival.hpp"
#include "statcpp/robust.hpp"
#include "statcpp/power_analysis.hpp"

// Module 6: Data Handling
#include "statcpp/data_wrangling.hpp"
#include "statcpp/missing_data.hpp"

// Module 8: Development Infrastructure
#include "statcpp/numerical_utils.hpp"
