/**
 * @file statcpp.hpp
 * @brief statcppライブラリのメインヘッダファイル
 *
 * 統計解析ライブラリの全モジュールをインクルードする統合ヘッダです。
 * このファイルをインクルードすることで、すべての統計機能にアクセスできます。
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

// Module 8: Development Infrastructure
#include "statcpp/numerical_utils.hpp"
