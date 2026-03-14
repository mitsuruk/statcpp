#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "statcpp/missing_data.hpp"

// ============================================================================
// Missing Pattern Analysis Tests
// ============================================================================

/**
 * @brief Tests missing data pattern analysis
 * @test Verifies that missing data patterns are correctly identified and summarized
 */
TEST(MissingPatternTest, AnalyzePatterns) {
    std::vector<std::vector<double>> data = {
        {1.0, 2.0, 3.0},
        {statcpp::NA, 2.0, 3.0},
        {1.0, statcpp::NA, 3.0},
        {1.0, 2.0, statcpp::NA},
        {1.0, 2.0, 3.0},
        {statcpp::NA, statcpp::NA, 3.0}
    };

    auto result = statcpp::analyze_missing_patterns(data);

    // Two complete cases
    EXPECT_EQ(result.n_complete_cases, 2);

    // Overall missing rate: 5 / 18 ≈ 0.278
    EXPECT_NEAR(result.overall_missing_rate, 5.0 / 18.0, 0.01);

    // Missing rate for each column
    EXPECT_NEAR(result.missing_rates[0], 2.0 / 6.0, 0.01);  // Column 0: 2/6
    EXPECT_NEAR(result.missing_rates[1], 2.0 / 6.0, 0.01);  // Column 1: 2/6
    EXPECT_NEAR(result.missing_rates[2], 1.0 / 6.0, 0.01);  // Column 2: 1/6
}

/**
 * @brief Tests missing pattern analysis with empty input data
 * @test Verifies that an exception is thrown when analyzing patterns of empty data
 */
TEST(MissingPatternTest, EmptyData) {
    std::vector<std::vector<double>> data;
    EXPECT_THROW(statcpp::analyze_missing_patterns(data), std::invalid_argument);
}

/**
 * @brief Tests missing pattern analysis with no missing values
 * @test Verifies that analysis correctly reports zero missing rate when no values are missing
 */
TEST(MissingPatternTest, NoMissing) {
    std::vector<std::vector<double>> data = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    auto result = statcpp::analyze_missing_patterns(data);

    EXPECT_EQ(result.n_complete_cases, 3);
    EXPECT_DOUBLE_EQ(result.overall_missing_rate, 0.0);
    EXPECT_EQ(result.n_patterns, 1);  // Only pattern without missing values
}

/**
 * @brief Tests missing pattern analysis when all values are missing
 * @test Verifies that analysis correctly reports 100% missing rate when all values are missing
 */
TEST(MissingPatternTest, AllMissing) {
    std::vector<std::vector<double>> data = {
        {statcpp::NA, statcpp::NA},
        {statcpp::NA, statcpp::NA}
    };

    auto result = statcpp::analyze_missing_patterns(data);

    EXPECT_EQ(result.n_complete_cases, 0);
    EXPECT_DOUBLE_EQ(result.overall_missing_rate, 1.0);
}

/**
 * @brief Tests creation of missing data indicator matrix
 * @test Verifies that binary indicator matrix correctly marks missing values as 1 and observed as 0
 */
TEST(MissingIndicatorTest, CreateIndicator) {
    std::vector<std::vector<double>> data = {
        {1.0, statcpp::NA, 3.0},
        {statcpp::NA, 2.0, statcpp::NA}
    };

    auto indicator = statcpp::create_missing_indicator(data);

    EXPECT_EQ(indicator.size(), 2);
    EXPECT_EQ(indicator[0].size(), 3);

    EXPECT_DOUBLE_EQ(indicator[0][0], 0.0);
    EXPECT_DOUBLE_EQ(indicator[0][1], 1.0);
    EXPECT_DOUBLE_EQ(indicator[0][2], 0.0);
    EXPECT_DOUBLE_EQ(indicator[1][0], 1.0);
    EXPECT_DOUBLE_EQ(indicator[1][1], 0.0);
    EXPECT_DOUBLE_EQ(indicator[1][2], 1.0);
}

// ============================================================================
// MCAR Test
// ============================================================================

/**
 * @brief Tests MCAR (Missing Completely At Random) test with randomly missing data
 * @test Verifies that MCAR test produces valid p-value and degrees of freedom
 */
TEST(MCARTest, SimpleMCARData) {
    // Generate MCAR data (insert missing values randomly)
    std::vector<std::vector<double>> data;

    // Two uncorrelated variables, random missing
    for (int i = 0; i < 100; ++i) {
        double x = static_cast<double>(i);
        double y = static_cast<double>(i % 10);

        // Insert missing values randomly (approximately 20%)
        if (i % 5 == 0) {
            x = statcpp::NA;
        }
        if (i % 7 == 0) {
            y = statcpp::NA;
        }

        data.push_back({x, y});
    }

    auto result = statcpp::test_mcar_simple(data);

    // Verify that p-value is computed
    EXPECT_GE(result.p_value, 0.0);
    EXPECT_LE(result.p_value, 1.0);

    // Verify that degrees of freedom is positive
    EXPECT_GE(result.df, 0);
}

/**
 * @brief Tests MCAR test with empty input data
 * @test Verifies that an exception is thrown when testing MCAR with empty data
 */
TEST(MCARTest, EmptyData) {
    std::vector<std::vector<double>> data;
    EXPECT_THROW(statcpp::test_mcar_simple(data), std::invalid_argument);
}

/**
 * @brief Tests diagnosis of missing data mechanism
 * @test Verifies that missing mechanism is classified as MCAR, MAR, MNAR, or unknown
 */
TEST(MCARTest, DiagnoseMechanism) {
    std::vector<std::vector<double>> data;

    for (int i = 0; i < 50; ++i) {
        double x = static_cast<double>(i);
        double y = static_cast<double>(i * 2);

        if (i % 5 == 0) {
            x = statcpp::NA;
        }

        data.push_back({x, y});
    }

    auto mechanism = statcpp::diagnose_missing_mechanism(data);

    // Verify that result is a valid enum value
    EXPECT_TRUE(mechanism == statcpp::missing_mechanism::mcar ||
                mechanism == statcpp::missing_mechanism::mar ||
                mechanism == statcpp::missing_mechanism::mnar ||
                mechanism == statcpp::missing_mechanism::unknown);
}

// ============================================================================
// Multiple Imputation Tests
// ============================================================================

/**
 * @brief Tests multiple imputation using Predictive Mean Matching (PMM)
 * @test Verifies that PMM generates multiple imputed datasets without missing values and computes pooled estimates
 */
TEST(MultipleImputationTest, PMMBasic) {
    std::vector<std::vector<double>> data = {
        {1.0, 2.0},
        {2.0, 4.0},
        {statcpp::NA, 6.0},
        {4.0, 8.0},
        {5.0, statcpp::NA},
        {6.0, 12.0}
    };

    auto result = statcpp::multiple_imputation_pmm(data, 5, 42);

    // 5 imputed datasets are generated
    EXPECT_EQ(result.m, 5);
    EXPECT_EQ(result.imputed_datasets.size(), 5);

    // Verify that each dataset has no NA values
    for (const auto& dataset : result.imputed_datasets) {
        for (const auto& row : dataset) {
            for (double val : row) {
                EXPECT_FALSE(statcpp::is_na(val));
            }
        }
    }

    // Verify that pooled estimates are computed
    EXPECT_EQ(result.pooled_means.size(), 2);
    EXPECT_EQ(result.pooled_vars.size(), 2);
    EXPECT_EQ(result.fraction_missing_info.size(), 2);
}

/**
 * @brief Tests multiple imputation using bootstrap method
 * @test Verifies that bootstrap imputation generates multiple complete datasets
 */
TEST(MultipleImputationTest, BootstrapBasic) {
    std::vector<std::vector<double>> data = {
        {1.0, 10.0},
        {2.0, 20.0},
        {statcpp::NA, 30.0},
        {4.0, statcpp::NA},
        {5.0, 50.0}
    };

    auto result = statcpp::multiple_imputation_bootstrap(data, 5, 42);

    EXPECT_EQ(result.m, 5);
    EXPECT_EQ(result.imputed_datasets.size(), 5);

    // Verify that each dataset has no NA values
    for (const auto& dataset : result.imputed_datasets) {
        for (const auto& row : dataset) {
            for (double val : row) {
                EXPECT_FALSE(statcpp::is_na(val));
            }
        }
    }
}

/**
 * @brief Tests multiple imputation with empty input data
 * @test Verifies that exceptions are thrown when attempting imputation on empty data
 */
TEST(MultipleImputationTest, EmptyData) {
    std::vector<std::vector<double>> data;
    EXPECT_THROW(statcpp::multiple_imputation_pmm(data), std::invalid_argument);
    EXPECT_THROW(statcpp::multiple_imputation_bootstrap(data), std::invalid_argument);
}

/**
 * @brief Tests multiple imputation when data has no missing values
 * @test Verifies that imputation works correctly and preserves data when no values are missing
 */
TEST(MultipleImputationTest, NoMissing) {
    std::vector<std::vector<double>> data = {
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0}
    };

    auto result = statcpp::multiple_imputation_pmm(data, 3, 42);

    // Works without error even when there is no missing data
    EXPECT_EQ(result.m, 3);

    // Data remains unchanged after imputation
    for (const auto& dataset : result.imputed_datasets) {
        EXPECT_EQ(dataset.size(), data.size());
        for (std::size_t i = 0; i < data.size(); ++i) {
            for (std::size_t j = 0; j < data[i].size(); ++j) {
                EXPECT_DOUBLE_EQ(dataset[i][j], data[i][j]);
            }
        }
    }
}

/**
 * @brief Tests Rubin's rules for pooling multiple imputation results
 * @test Verifies that within-imputation and between-imputation variances are computed and pooled correctly
 */
TEST(MultipleImputationTest, RubinsRules) {
    // Verify that pooling by Rubin's rules works correctly
    std::vector<std::vector<double>> data = {
        {1.0, 2.0},
        {2.0, 4.0},
        {3.0, 6.0},
        {statcpp::NA, 8.0},
        {5.0, 10.0}
    };

    auto result = statcpp::multiple_imputation_pmm(data, 10, 42);

    // Within-imputation and between-imputation variances are computed
    EXPECT_EQ(result.within_vars.size(), 2);
    EXPECT_EQ(result.between_vars.size(), 2);

    // Pooled variance = W + (1 + 1/m) * B
    for (std::size_t j = 0; j < 2; ++j) {
        double expected_pooled = result.within_vars[j] +
                                 (1.0 + 1.0 / static_cast<double>(result.m)) *
                                 result.between_vars[j];
        EXPECT_NEAR(result.pooled_vars[j], expected_pooled, 1e-10);
    }
}

/**
 * @brief Tests FMI (Fraction of Missing Information) calculation
 * @test Verifies that FMI is within valid range [0, 1] and calculated correctly
 *       FMI = ((1 + 1/m) * B) / T where T is pooled variance (Rubin, 1987)
 */
TEST(MultipleImputationTest, FMIValidation) {
    // Data with missing values
    std::vector<std::vector<double>> data = {
        {1.0, 2.0},
        {2.0, 4.0},
        {3.0, 6.0},
        {statcpp::NA, 8.0},
        {5.0, statcpp::NA},
        {6.0, 12.0}
    };

    auto result = statcpp::multiple_imputation_pmm(data, 20, 42);

    // FMI is computed
    EXPECT_EQ(result.fraction_missing_info.size(), 2);

    // FMI should be within [0, 1] range
    for (std::size_t j = 0; j < 2; ++j) {
        EXPECT_GE(result.fraction_missing_info[j], 0.0)
            << "FMI should be >= 0 for variable " << j;
        EXPECT_LE(result.fraction_missing_info[j], 1.0)
            << "FMI should be <= 1 for variable " << j;
    }

    // Verify FMI formula: FMI = ((1 + 1/m) * B) / T
    for (std::size_t j = 0; j < 2; ++j) {
        if (result.pooled_vars[j] > 1e-10) {
            double expected_fmi = (1.0 + 1.0 / static_cast<double>(result.m)) *
                                  result.between_vars[j] / result.pooled_vars[j];
            EXPECT_NEAR(result.fraction_missing_info[j], expected_fmi, 1e-10)
                << "FMI calculation mismatch for variable " << j;
        }
    }
}

/**
 * @brief Tests FMI with Bootstrap multiple imputation
 * @test Verifies that FMI from bootstrap imputation is also within valid range
 */
TEST(MultipleImputationTest, FMIValidationBootstrap) {
    std::vector<std::vector<double>> data = {
        {1.0, 2.0},
        {2.0, 4.0},
        {statcpp::NA, 6.0},
        {4.0, statcpp::NA},
        {5.0, 10.0}
    };

    auto result = statcpp::multiple_imputation_bootstrap(data, 15, 123);

    // FMI should be within [0, 1] range
    for (std::size_t j = 0; j < result.fraction_missing_info.size(); ++j) {
        EXPECT_GE(result.fraction_missing_info[j], 0.0)
            << "Bootstrap FMI should be >= 0 for variable " << j;
        EXPECT_LE(result.fraction_missing_info[j], 1.0)
            << "Bootstrap FMI should be <= 1 for variable " << j;
    }
}

// ============================================================================
// Sensitivity Analysis Tests
// ============================================================================

/**
 * @brief Tests sensitivity analysis using pattern mixture model
 * @test Verifies that sensitivity analysis explores different delta values and their impact on estimates
 */
TEST(SensitivityAnalysisTest, PatternMixture) {
    std::vector<double> data = {1.0, 2.0, 3.0, statcpp::NA, 5.0, statcpp::NA};
    std::vector<double> deltas = {-2.0, -1.0, 0.0, 1.0, 2.0};

    auto result = statcpp::sensitivity_analysis_pattern_mixture(data, deltas);

    EXPECT_EQ(result.delta_values.size(), 5);
    EXPECT_EQ(result.estimated_means.size(), 5);
    EXPECT_EQ(result.estimated_vars.size(), 5);

    // delta=0 assumes MAR (uses observed mean as is)
    // Mean of observed values: (1+2+3+5)/4 = 2.75
    double obs_mean = 2.75;
    double prop_obs = 4.0 / 6.0;
    double prop_miss = 2.0 / 6.0;

    // Estimated mean at delta=0 should equal observed mean
    EXPECT_NEAR(result.estimated_means[2], obs_mean * prop_obs + obs_mean * prop_miss, 0.01);

    // Estimated mean increases as delta increases
    for (std::size_t i = 1; i < result.estimated_means.size(); ++i) {
        EXPECT_GT(result.estimated_means[i], result.estimated_means[i - 1]);
    }
}

/**
 * @brief Tests sensitivity analysis using selection model
 * @test Verifies that selection model sensitivity analysis explores different phi values
 */
TEST(SensitivityAnalysisTest, SelectionModel) {
    std::vector<double> data = {1.0, 2.0, 3.0, statcpp::NA, 5.0};
    std::vector<double> phis = {-1.0, 0.0, 1.0};

    auto result = statcpp::sensitivity_analysis_selection_model(data, phis);

    EXPECT_EQ(result.delta_values.size(), 3);
    EXPECT_EQ(result.estimated_means.size(), 3);

    // phi=0 assumes MAR
    // phi>0 suggests missing values tend to be lower
}

/**
 * @brief Tests sensitivity analysis with empty input data
 * @test Verifies that exceptions are thrown when performing sensitivity analysis on empty data
 */
TEST(SensitivityAnalysisTest, EmptyData) {
    std::vector<double> data;
    std::vector<double> deltas = {0.0};

    EXPECT_THROW(statcpp::sensitivity_analysis_pattern_mixture(data, deltas),
                 std::invalid_argument);
    EXPECT_THROW(statcpp::sensitivity_analysis_selection_model(data, deltas),
                 std::invalid_argument);
}

/**
 * @brief Tests sensitivity analysis when all values are missing
 * @test Verifies that an exception is thrown when all observed values are missing
 */
TEST(SensitivityAnalysisTest, AllMissing) {
    std::vector<double> data = {statcpp::NA, statcpp::NA, statcpp::NA};
    std::vector<double> deltas = {0.0};

    EXPECT_THROW(statcpp::sensitivity_analysis_pattern_mixture(data, deltas),
                 std::invalid_argument);
}

// ============================================================================
// Tipping Point Analysis Tests
// ============================================================================

/**
 * @brief Tests finding the tipping point for inference robustness
 * @test Verifies that tipping point analysis identifies the delta value where conclusions change
 */
TEST(TippingPointTest, FindTippingPoint) {
    // When observed mean is positive and searching for threshold 0
    std::vector<double> data = {1.0, 2.0, 3.0, statcpp::NA, statcpp::NA};

    auto result = statcpp::find_tipping_point(data, 0.0, -10.0, 10.0, 100);

    // Observed mean is 2.0, missing rate is 2/5
    // delta < 0 should be able to make overall mean <= 0
    if (result.found) {
        EXPECT_LT(result.tipping_point, 0.0);
    }
}

/**
 * @brief Tests tipping point analysis when no tipping point exists in the search range
 * @test Verifies that analysis correctly reports when no tipping point is found within the specified range
 */
TEST(TippingPointTest, NoTippingPoint) {
    // When all observed values are positive and threshold is very high
    std::vector<double> data = {1.0, 2.0, 3.0, statcpp::NA};

    auto result = statcpp::find_tipping_point(data, 100.0, -5.0, 5.0, 50);

    // No tipping point found within the specified range
    EXPECT_FALSE(result.found);
}

// ============================================================================
// Complete Case Analysis Tests
// ============================================================================

/**
 * @brief Tests extraction of complete cases (listwise deletion)
 * @test Verifies that rows with any missing values are removed and statistics are computed correctly
 */
TEST(CompleteCaseTest, ExtractCompleteCases) {
    std::vector<std::vector<double>> data = {
        {1.0, 2.0, 3.0},
        {statcpp::NA, 2.0, 3.0},
        {1.0, 2.0, 3.0},
        {1.0, statcpp::NA, 3.0}
    };

    auto result = statcpp::extract_complete_cases(data);

    EXPECT_EQ(result.n_complete, 2);
    EXPECT_EQ(result.n_dropped, 2);
    EXPECT_DOUBLE_EQ(result.proportion_complete, 0.5);

    // Verify complete case data
    EXPECT_EQ(result.complete_data.size(), 2);
    for (const auto& row : result.complete_data) {
        for (double val : row) {
            EXPECT_FALSE(statcpp::is_na(val));
        }
    }
}

/**
 * @brief Tests complete case extraction when all cases are complete
 * @test Verifies that all rows are retained when no missing values exist
 */
TEST(CompleteCaseTest, AllComplete) {
    std::vector<std::vector<double>> data = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    auto result = statcpp::extract_complete_cases(data);

    EXPECT_EQ(result.n_complete, 2);
    EXPECT_EQ(result.n_dropped, 0);
    EXPECT_DOUBLE_EQ(result.proportion_complete, 1.0);
}

/**
 * @brief Tests complete case extraction when all cases have missing values
 * @test Verifies that no rows are retained when every row contains at least one missing value
 */
TEST(CompleteCaseTest, AllIncomplete) {
    std::vector<std::vector<double>> data = {
        {statcpp::NA, 2.0},
        {1.0, statcpp::NA}
    };

    auto result = statcpp::extract_complete_cases(data);

    EXPECT_EQ(result.n_complete, 0);
    EXPECT_EQ(result.n_dropped, 2);
    EXPECT_DOUBLE_EQ(result.proportion_complete, 0.0);
}

// ============================================================================
// Pairwise Correlation Tests
// ============================================================================

/**
 * @brief Tests pairwise correlation matrix computation with missing data
 * @test Verifies that correlation matrix uses all available pairs and maintains symmetry
 */
TEST(PairwiseCorrelationTest, Basic) {
    std::vector<std::vector<double>> data = {
        {1.0, 2.0, 3.0},
        {2.0, 4.0, 6.0},
        {3.0, statcpp::NA, 9.0},
        {4.0, 8.0, statcpp::NA},
        {5.0, 10.0, 15.0}
    };

    auto corr_matrix = statcpp::correlation_matrix_pairwise(data);

    EXPECT_EQ(corr_matrix.size(), 3);
    EXPECT_EQ(corr_matrix[0].size(), 3);

    // Diagonal elements are 1
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(corr_matrix[i][i], 1.0);
    }

    // Symmetry
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            if (!statcpp::is_na(corr_matrix[i][j])) {
                EXPECT_DOUBLE_EQ(corr_matrix[i][j], corr_matrix[j][i]);
            }
        }
    }
}

/**
 * @brief Tests pairwise correlation with empty input data
 * @test Verifies that an empty correlation matrix is returned for empty data
 */
TEST(PairwiseCorrelationTest, EmptyData) {
    std::vector<std::vector<double>> data;
    auto corr_matrix = statcpp::correlation_matrix_pairwise(data);
    EXPECT_TRUE(corr_matrix.empty());
}

/**
 * @brief Tests pairwise correlation with perfectly correlated variables
 * @test Verifies that perfect linear correlation is detected correctly (correlation = 1.0)
 */
TEST(PairwiseCorrelationTest, PerfectCorrelation) {
    std::vector<std::vector<double>> data = {
        {1.0, 2.0},
        {2.0, 4.0},
        {3.0, 6.0},
        {4.0, 8.0}
    };

    auto corr_matrix = statcpp::correlation_matrix_pairwise(data);

    // Perfect correlation
    EXPECT_NEAR(corr_matrix[0][1], 1.0, 1e-10);
}

// ============================================================================
// Conditional Mean Imputation Tests
// ============================================================================

/**
 * @brief Tests conditional mean imputation using predictor variables
 * @test Verifies that missing values are imputed based on regression from predictor variables
 */
TEST(ConditionalMeanTest, Basic) {
    std::vector<std::vector<double>> data = {
        {1.0, 2.0},
        {2.0, 4.0},
        {3.0, 6.0},
        {statcpp::NA, 8.0},
        {5.0, 10.0}
    };

    auto imputed = statcpp::impute_conditional_mean(data, 0, {1});

    EXPECT_EQ(imputed.size(), 5);

    // Non-missing values remain unchanged
    EXPECT_DOUBLE_EQ(imputed[0], 1.0);
    EXPECT_DOUBLE_EQ(imputed[1], 2.0);
    EXPECT_DOUBLE_EQ(imputed[2], 3.0);
    EXPECT_DOUBLE_EQ(imputed[4], 5.0);

    // Missing value is imputed
    EXPECT_FALSE(statcpp::is_na(imputed[3]));

    // Since Y = X/2 relationship exists, Y approximately 4 is expected for X=8
    EXPECT_NEAR(imputed[3], 4.0, 0.5);
}

/**
 * @brief Tests conditional mean imputation without predictor variables
 * @test Verifies that unconditional mean imputation is used when no predictors are specified
 */
TEST(ConditionalMeanTest, NoPredictors) {
    std::vector<std::vector<double>> data = {
        {1.0},
        {2.0},
        {statcpp::NA},
        {4.0}
    };

    auto imputed = statcpp::impute_conditional_mean(data, 0, {});

    // Impute with mean when there are no predictor variables
    double expected_mean = (1.0 + 2.0 + 4.0) / 3.0;
    EXPECT_NEAR(imputed[2], expected_mean, 0.01);
}
