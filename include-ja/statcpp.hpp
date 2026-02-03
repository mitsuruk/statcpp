/**
 * @file statcpp.hpp
 * @brief statcppライブラリのメインヘッダファイル
 *
 * 統計解析ライブラリの全モジュールをインクルードする統合ヘッダです。
 * このファイルをインクルードすることで、すべての統計機能にアクセスできます。
 */

#pragma once

// Module 1: Descriptive Statistics
#include "basic_statistics.hpp"
#include "order_statistics.hpp"
#include "dispersion_spread.hpp"
#include "shape_of_distribution.hpp"
#include "correlation_covariance.hpp"
#include "frequency_distribution.hpp"

// Module 2: Probability Distributions
#include "special_functions.hpp"
#include "random_engine.hpp"
#include "continuous_distributions.hpp"
#include "discrete_distributions.hpp"

// Module 3: Inferential Statistics
#include "estimation.hpp"
#include "parametric_tests.hpp"
#include "nonparametric_tests.hpp"
#include "effect_size.hpp"
#include "resampling.hpp"

// Module 4: Statistical Modeling
#include "linear_regression.hpp"
#include "anova.hpp"
#include "glm.hpp"
#include "model_selection.hpp"

// Module 5: Applied Analysis
#include "distance_metrics.hpp"

// Module 8: Development Infrastructure
#include "numerical_utils.hpp"
