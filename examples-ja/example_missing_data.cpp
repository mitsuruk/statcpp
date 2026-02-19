/**
 * @file example_missing_data.cpp
 * @brief 欠損データ分析のサンプルコード
 *
 * 欠損パターン分析、MCAR検定、多重代入法、感度分析、
 * ティッピングポイント分析等の欠損データ処理手法の使用例を示します。
 */

#include <iostream>
#include <iomanip>
#include "statcpp/missing_data.hpp"

int main() {
    std::cout << "=== 欠損データ分析の例 ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    // ============================================================================
    // 1. 欠損パターンの分析
    // ============================================================================
    std::cout << "\n1. 欠損パターンの分析" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<std::vector<double>> data = {
        {1.0, 2.0, 3.0},
        {statcpp::NA, 2.0, 3.0},
        {1.0, statcpp::NA, 3.0},
        {1.0, 2.0, statcpp::NA},
        {1.0, 2.0, 3.0},
        {statcpp::NA, statcpp::NA, 3.0},
        {4.0, 5.0, 6.0}
    };

    auto pattern_info = statcpp::analyze_missing_patterns(data);

    std::cout << "完全ケース数: " << pattern_info.n_complete_cases << std::endl;
    std::cout << "全体の欠損率: " << pattern_info.overall_missing_rate << std::endl;
    std::cout << "ユニークなパターン数: " << pattern_info.n_patterns << std::endl;

    std::cout << "\n変数ごとの欠損率:" << std::endl;
    for (std::size_t i = 0; i < pattern_info.missing_rates.size(); ++i) {
        std::cout << "  変数 " << i << ": " << pattern_info.missing_rates[i] << std::endl;
    }

    std::cout << "\n欠損パターン:" << std::endl;
    for (std::size_t i = 0; i < pattern_info.patterns.size(); ++i) {
        std::cout << "  パターン " << i << ": [";
        for (std::size_t j = 0; j < pattern_info.patterns[i].size(); ++j) {
            std::cout << static_cast<int>(pattern_info.patterns[i][j]);
            if (j + 1 < pattern_info.patterns[i].size()) std::cout << ", ";
        }
        std::cout << "] - 件数: " << pattern_info.pattern_counts[i] << std::endl;
    }

    // ============================================================================
    // 2. MCAR検定
    // ============================================================================
    std::cout << "\n2. LittleのMCAR検定（簡易版）" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<std::vector<double>> test_data;
    for (int i = 0; i < 100; ++i) {
        double x = static_cast<double>(i);
        double y = static_cast<double>(i % 10);

        // ランダムに欠損を挿入
        if (i % 5 == 0) x = statcpp::NA;
        if (i % 7 == 0) y = statcpp::NA;

        test_data.push_back({x, y});
    }

    auto mcar_result = statcpp::test_mcar_simple(test_data);

    std::cout << "カイ二乗統計量: " << mcar_result.chi_square << std::endl;
    std::cout << "自由度: " << mcar_result.df << std::endl;
    std::cout << "P値: " << mcar_result.p_value << std::endl;
    std::cout << "MCARか? " << (mcar_result.is_mcar ? "はい" : "いいえ") << std::endl;
    std::cout << "解釈: " << mcar_result.interpretation << std::endl;

    // ============================================================================
    // 3. 欠損メカニズムの診断
    // ============================================================================
    std::cout << "\n3. 欠損メカニズムの診断" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    auto mechanism = statcpp::diagnose_missing_mechanism(test_data);

    std::cout << "検出されたメカニズム: ";
    switch (mechanism) {
        case statcpp::missing_mechanism::mcar:
            std::cout << "MCAR（完全にランダムな欠損）" << std::endl;
            break;
        case statcpp::missing_mechanism::mar:
            std::cout << "MAR（ランダムな欠損）" << std::endl;
            break;
        case statcpp::missing_mechanism::mnar:
            std::cout << "MNAR（非ランダムな欠損）" << std::endl;
            break;
        case statcpp::missing_mechanism::unknown:
            std::cout << "不明" << std::endl;
            break;
    }

    // ============================================================================
    // 4. 多重代入（PMM法）
    // ============================================================================
    std::cout << "\n4. 多重代入法（PMM）" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<std::vector<double>> impute_data = {
        {1.0, 2.0},
        {2.0, 4.0},
        {statcpp::NA, 6.0},
        {4.0, 8.0},
        {5.0, statcpp::NA},
        {6.0, 12.0}
    };

    auto mi_result = statcpp::multiple_imputation_pmm(impute_data, 5, 42);

    std::cout << "代入回数: " << mi_result.m << std::endl;
    std::cout << "\nプールされた平均:" << std::endl;
    for (std::size_t i = 0; i < mi_result.pooled_means.size(); ++i) {
        std::cout << "  変数 " << i << ": " << mi_result.pooled_means[i] << std::endl;
    }

    std::cout << "\nプールされた分散:" << std::endl;
    for (std::size_t i = 0; i < mi_result.pooled_vars.size(); ++i) {
        std::cout << "  変数 " << i << ": " << mi_result.pooled_vars[i] << std::endl;
    }

    std::cout << "\n欠損情報割合（FMI）:" << std::endl;
    for (std::size_t i = 0; i < mi_result.fraction_missing_info.size(); ++i) {
        std::cout << "  変数 " << i << ": " << mi_result.fraction_missing_info[i] << std::endl;
    }

    // ============================================================================
    // 5. 感度分析（パターン混合モデル）
    // ============================================================================
    std::cout << "\n5. 感度分析（パターン混合モデル）" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> sens_data = {1.0, 2.0, 3.0, statcpp::NA, 5.0, statcpp::NA};
    std::vector<double> delta_values = {-2.0, -1.0, 0.0, 1.0, 2.0};

    auto sens_result = statcpp::sensitivity_analysis_pattern_mixture(sens_data, delta_values);

    std::cout << "元の平均推定値: " << sens_result.original_mean << std::endl;
    std::cout << "\n異なる仮定下での推定平均:" << std::endl;
    for (std::size_t i = 0; i < sens_result.delta_values.size(); ++i) {
        std::cout << "  delta = " << std::setw(5) << sens_result.delta_values[i]
                  << " -> 平均 = " << sens_result.estimated_means[i] << std::endl;
    }

    // ============================================================================
    // 6. ティッピングポイント分析
    // ============================================================================
    std::cout << "\n6. ティッピングポイント分析" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> tip_data = {1.0, 2.0, 3.0, statcpp::NA, statcpp::NA};
    auto tip_result = statcpp::find_tipping_point(tip_data, 0.0, -10.0, 10.0, 100);

    if (tip_result.found) {
        std::cout << "ティッピングポイントが見つかりました!" << std::endl;
        std::cout << "  Delta値: " << tip_result.tipping_point << std::endl;
        std::cout << "  閾値: " << tip_result.threshold << std::endl;
    } else {
        std::cout << "指定範囲内にティッピングポイントが見つかりませんでした。" << std::endl;
    }
    std::cout << "解釈: " << tip_result.interpretation << std::endl;

    // ============================================================================
    // 7. 完全ケース分析
    // ============================================================================
    std::cout << "\n7. 完全ケース分析" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    auto cc_result = statcpp::extract_complete_cases(data);

    std::cout << "完全ケース数: " << cc_result.n_complete << std::endl;
    std::cout << "除外ケース数: " << cc_result.n_dropped << std::endl;
    std::cout << "完全ケース割合: " << cc_result.proportion_complete << std::endl;

    std::cout << "\n完全ケースデータ:" << std::endl;
    for (const auto& row : cc_result.complete_data) {
        std::cout << "  [";
        for (std::size_t i = 0; i < row.size(); ++i) {
            std::cout << row[i];
            if (i + 1 < row.size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // ============================================================================
    // 8. ペアワイズ相関（欠損値がある場合）
    // ============================================================================
    std::cout << "\n8. ペアワイズ相関行列" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<std::vector<double>> corr_data = {
        {1.0, 2.0, 3.0},
        {2.0, 4.0, 6.0},
        {3.0, statcpp::NA, 9.0},
        {4.0, 8.0, statcpp::NA},
        {5.0, 10.0, 15.0}
    };

    auto corr_matrix = statcpp::correlation_matrix_pairwise(corr_data);

    std::cout << "相関行列（ペアワイズ除外）:" << std::endl;
    for (std::size_t i = 0; i < corr_matrix.size(); ++i) {
        std::cout << "  ";
        for (std::size_t j = 0; j < corr_matrix[i].size(); ++j) {
            if (statcpp::is_na(corr_matrix[i][j])) {
                std::cout << std::setw(8) << "NA";
            } else {
                std::cout << std::setw(8) << corr_matrix[i][j];
            }
        }
        std::cout << std::endl;
    }

    std::cout << "\n=== 例の実行が正常に完了しました ===" << std::endl;

    return 0;
}
