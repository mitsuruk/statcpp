/**
 * @file example_robust.cpp
 * @brief ロバスト統計のサンプルコード
 *
 * MAD（中央絶対偏差）、外れ値検出（IQR法、Z-score法）、
 * ウィンザー化、Hodges-Lehmann推定量等のロバスト統計手法の使用例を示します。
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include "statcpp/robust.hpp"
#include "statcpp/dispersion_spread.hpp"

int main() {
    std::cout << "=== ロバスト統計のサンプル ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    // ============================================================================
    // 1. MAD（中央絶対偏差）
    // ============================================================================
    std::cout << "\n1. MAD（中央絶対偏差）" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    double median_val = statcpp::median(data.begin(), data.end());
    double mad_val = statcpp::mad(data.begin(), data.end());
    double mad_scaled_val = statcpp::mad_scaled(data.begin(), data.end());
    double stddev_val = statcpp::sample_stddev(data.begin(), data.end());

    std::cout << "データ: [";
    for (std::size_t i = 0; i < data.size(); ++i) {
        std::cout << data[i];
        if (i + 1 < data.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "\n散らばりの指標:" << std::endl;
    std::cout << "  中央値: " << median_val << std::endl;
    std::cout << "  MAD: " << mad_val << std::endl;
    std::cout << "  MAD (スケール調整済み): " << mad_scaled_val << std::endl;
    std::cout << "  標準偏差: " << stddev_val << std::endl;

    std::cout << "\n注: 正規分布データでは、スケール調整済みMAD ≈ 標準偏差" << std::endl;
    std::cout << "    スケール係数 = 1.4826" << std::endl;

    // 外れ値を含むデータでのMADとSDの比較
    std::vector<double> data_with_outlier = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0};

    double mad_scaled_outlier = statcpp::mad_scaled(data_with_outlier.begin(), data_with_outlier.end());
    double stddev_outlier = statcpp::sample_stddev(data_with_outlier.begin(), data_with_outlier.end());

    std::cout << "\n外れ値 (100) を含む場合:" << std::endl;
    std::cout << "  MAD (スケール調整済み): " << mad_scaled_outlier << std::endl;
    std::cout << "  標準偏差: " << stddev_outlier << std::endl;
    std::cout << "  MADはロバスト: 変化量 " << (mad_scaled_outlier - mad_scaled_val) << std::endl;
    std::cout << "  標準偏差は非ロバスト: 変化量 " << (stddev_outlier - stddev_val) << std::endl;

    // ============================================================================
    // 2. 外れ値検出（IQR法）
    // ============================================================================
    std::cout << "\n2. 外れ値検出（IQR法 - Tukeyの囲い）" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> test_data = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 50, 55  // 50 と 55 は外れ値
    };

    auto iqr_result = statcpp::detect_outliers_iqr(test_data.begin(), test_data.end(), 1.5);

    std::cout << "四分位数:" << std::endl;
    std::cout << "  Q1: " << iqr_result.q1 << std::endl;
    std::cout << "  Q3: " << iqr_result.q3 << std::endl;
    std::cout << "  IQR: " << iqr_result.iqr_value << std::endl;

    std::cout << "\n囲い (k=1.5):" << std::endl;
    std::cout << "  下側囲い: " << iqr_result.lower_fence << std::endl;
    std::cout << "  上側囲い: " << iqr_result.upper_fence << std::endl;

    std::cout << "\n検出された外れ値 (" << iqr_result.outliers.size() << "個):" << std::endl;
    for (std::size_t i = 0; i < iqr_result.outliers.size(); ++i) {
        std::cout << "  インデックス " << iqr_result.outlier_indices[i]
                  << ": 値 = " << iqr_result.outliers[i] << std::endl;
    }

    // 極端な外れ値検出（k=3.0）
    auto extreme_outliers = statcpp::detect_outliers_iqr(test_data.begin(), test_data.end(), 3.0);
    std::cout << "\n極端な外れ値 (k=3.0): " << extreme_outliers.outliers.size() << "個" << std::endl;

    // ============================================================================
    // 3. Z-score による外れ値検出
    // ============================================================================
    std::cout << "\n3. 外れ値検出（Z-score法）" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    auto zscore_result = statcpp::detect_outliers_zscore(test_data.begin(), test_data.end(), 3.0);

    std::cout << "閾値 (Z = 3.0):" << std::endl;
    std::cout << "  下限: " << zscore_result.lower_fence << std::endl;
    std::cout << "  上限: " << zscore_result.upper_fence << std::endl;

    std::cout << "\n検出された外れ値 (" << zscore_result.outliers.size() << "個):" << std::endl;
    for (std::size_t i = 0; i < zscore_result.outliers.size(); ++i) {
        std::cout << "  インデックス " << zscore_result.outlier_indices[i]
                  << ": 値 = " << zscore_result.outliers[i] << std::endl;
    }

    // ============================================================================
    // 4. Modified Z-score による外れ値検出
    // ============================================================================
    std::cout << "\n4. 外れ値検出（Modified Z-score法 - MADベース）" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    auto modified_zscore_result = statcpp::detect_outliers_modified_zscore(
        test_data.begin(), test_data.end(), 3.5
    );

    std::cout << "Modified Z-score法 (よりロバスト):" << std::endl;
    std::cout << "  検出された外れ値: " << modified_zscore_result.outliers.size() << "個" << std::endl;
    for (std::size_t i = 0; i < modified_zscore_result.outliers.size(); ++i) {
        std::cout << "  インデックス " << modified_zscore_result.outlier_indices[i]
                  << ": 値 = " << modified_zscore_result.outliers[i] << std::endl;
    }

    // ============================================================================
    // 5. 外れ値検出手法の比較
    // ============================================================================
    std::cout << "\n5. 外れ値検出手法の比較" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::cout << "検出された外れ値の数:" << std::endl;
    std::cout << "  IQR法 (k=1.5): " << iqr_result.outliers.size() << "個" << std::endl;
    std::cout << "  Z-score法 (閾値=3.0): " << zscore_result.outliers.size() << "個" << std::endl;
    std::cout << "  Modified Z-score法 (閾値=3.5): " << modified_zscore_result.outliers.size() << "個" << std::endl;

    std::cout << "\n推奨事項:" << std::endl;
    std::cout << "  - IQR法: 歪んだ分布に最適" << std::endl;
    std::cout << "  - Z-score法: 正規分布を仮定、外れ値の影響を受けやすい" << std::endl;
    std::cout << "  - Modified Z-score法: ロバスト、非正規データに適している" << std::endl;

    // ============================================================================
    // 6. ウィンザー化（Winsorization）
    // ============================================================================
    std::cout << "\n6. ウィンザー化" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> winsor_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 100};

    auto winsorized = statcpp::winsorize(winsor_data.begin(), winsor_data.end(), 0.1);

    std::cout << "元のデータ: [";
    for (std::size_t i = 0; i < winsor_data.size(); ++i) {
        std::cout << winsor_data[i];
        if (i + 1 < winsor_data.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "ウィンザー化後 (10%): [";
    for (std::size_t i = 0; i < winsorized.size(); ++i) {
        std::cout << winsorized[i];
        if (i + 1 < winsorized.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    double mean_original = statcpp::mean(winsor_data.begin(), winsor_data.end());
    double mean_winsorized = statcpp::mean(winsorized.begin(), winsorized.end());

    std::cout << "\nウィンザー化前の平均: " << mean_original << std::endl;
    std::cout << "ウィンザー化後の平均: " << mean_winsorized << std::endl;
    std::cout << "変化: " << (mean_original - mean_winsorized) << std::endl;

    std::cout << "\n注: ウィンザー化は極端な値を削除するのではなく、" << std::endl;
    std::cout << "    より極端でない値で置き換えます（トリミングとは異なる）。" << std::endl;

    // ============================================================================
    // 7. Cook's Distance（回帰診断）
    // ============================================================================
    std::cout << "\n7. Cook's Distance（回帰診断）" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    // 回帰分析の仮想データ
    std::vector<double> residuals = {0.5, -0.3, 0.2, -0.1, 0.4, -0.2, 5.0, 0.3};
    std::vector<double> hat_values = {0.15, 0.12, 0.18, 0.10, 0.14, 0.11, 0.25, 0.13};
    double mse = 1.5;
    std::size_t p = 2;  // 切片 + 1変数

    auto cooks_d = statcpp::cooks_distance(residuals, hat_values, mse, p);

    std::cout << "Cook's Distanceの値:" << std::endl;
    for (std::size_t i = 0; i < cooks_d.size(); ++i) {
        std::cout << "  観測 " << (i + 1) << ": D = " << cooks_d[i];
        if (cooks_d[i] > 1.0) {
            std::cout << " (影響力大)";
        } else if (cooks_d[i] > 0.5) {
            std::cout << " (影響力の可能性あり)";
        }
        std::cout << std::endl;
    }

    std::cout << "\n解釈:" << std::endl;
    std::cout << "  D > 1.0: 影響力の高い観測" << std::endl;
    std::cout << "  D > 0.5: 影響力の可能性あり" << std::endl;
    std::cout << "  D > 4/n: 一般的な目安 (ここでは: " << (4.0 / cooks_d.size()) << ")" << std::endl;

    // ============================================================================
    // 8. DFFITS（回帰診断）
    // ============================================================================
    std::cout << "\n8. DFFITS（回帰診断）" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    auto dffits_vals = statcpp::dffits(residuals, hat_values, mse);

    std::cout << "DFFITSの値:" << std::endl;
    double dffits_cutoff = 2.0 * std::sqrt(static_cast<double>(p) / cooks_d.size());
    std::cout << "カットオフ: ±" << dffits_cutoff << std::endl;

    for (std::size_t i = 0; i < dffits_vals.size(); ++i) {
        std::cout << "  観測 " << (i + 1) << ": DFFITS = " << dffits_vals[i];
        if (std::abs(dffits_vals[i]) > dffits_cutoff) {
            std::cout << " (影響力大)";
        }
        std::cout << std::endl;
    }

    std::cout << "\n解釈:" << std::endl;
    std::cout << "  |DFFITS| > 2*sqrt(p/n): 影響力のある観測" << std::endl;

    // ============================================================================
    // 9. Hodges-Lehmann 推定量
    // ============================================================================
    std::cout << "\n9. Hodges-Lehmann推定量（ロバストな位置推定）" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> location_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0};

    double mean_loc = statcpp::mean(location_data.begin(), location_data.end());
    double median_loc = statcpp::median(location_data.begin(), location_data.end());
    double hl_loc = statcpp::hodges_lehmann(location_data.begin(), location_data.end());

    std::cout << "外れ値を含むデータ: [1, 2, ..., 9, 100]" << std::endl;
    std::cout << "\n位置推定量:" << std::endl;
    std::cout << "  平均: " << mean_loc << " (外れ値の影響を受ける)" << std::endl;
    std::cout << "  中央値: " << median_loc << " (ロバスト)" << std::endl;
    std::cout << "  Hodges-Lehmann: " << hl_loc << " (ロバスト)" << std::endl;

    std::cout << "\n注: Hodges-Lehmannは全ペアの平均値の中央値です。" << std::endl;
    std::cout << "    対称分布では中央値よりも効率的です。" << std::endl;

    // ============================================================================
    // 10. Biweight Midvariance（ロバストな分散推定量）
    // ============================================================================
    std::cout << "\n10. Biweight Midvariance（ロバストな分散推定）" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> variance_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    double sample_var = statcpp::sample_variance(variance_data.begin(), variance_data.end());
    double bw_midvar = statcpp::biweight_midvariance(variance_data.begin(), variance_data.end());

    std::cout << "クリーンなデータ (外れ値なし):" << std::endl;
    std::cout << "  標本分散: " << sample_var << std::endl;
    std::cout << "  Biweight midvariance: " << bw_midvar << std::endl;

    // 外れ値を追加
    std::vector<double> variance_data_outlier = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0};

    double sample_var_out = statcpp::sample_variance(variance_data_outlier.begin(), variance_data_outlier.end());
    double bw_midvar_out = statcpp::biweight_midvariance(variance_data_outlier.begin(), variance_data_outlier.end());

    std::cout << "\n外れ値 (100) を含む場合:" << std::endl;
    std::cout << "  標本分散: " << sample_var_out << " (膨張)" << std::endl;
    std::cout << "  Biweight midvariance: " << bw_midvar_out << " (ロバスト)" << std::endl;

    std::cout << "\n外れ値による変化:" << std::endl;
    std::cout << "  標本分散: +" << (sample_var_out - sample_var) << std::endl;
    std::cout << "  Biweight midvariance: +" << (bw_midvar_out - bw_midvar) << std::endl;

    // ============================================================================
    // 11. ロバスト統計の重要性
    // ============================================================================
    std::cout << "\n11. まとめ: ロバスト統計の重要性" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::cout << "古典的推定量 vs ロバスト推定量:" << std::endl;
    std::cout << "\n位置:" << std::endl;
    std::cout << "  古典的: 平均 (外れ値に敏感)" << std::endl;
    std::cout << "  ロバスト: 中央値、Hodges-Lehmann (外れ値に頑健)" << std::endl;

    std::cout << "\nスケール（散らばり）:" << std::endl;
    std::cout << "  古典的: 標準偏差 (敏感)" << std::endl;
    std::cout << "  ロバスト: MAD、Biweight Midvariance (頑健)" << std::endl;

    std::cout << "\n外れ値検出:" << std::endl;
    std::cout << "  非ロバスト: Z-score法 (平均と標準偏差を使用)" << std::endl;
    std::cout << "  ロバスト: IQR法、Modified Z-score法 (中央値とMADを使用)" << std::endl;

    std::cout << "\nロバスト手法を使用すべき場合:" << std::endl;
    std::cout << "  - データに外れ値や測定誤差が含まれる可能性がある" << std::endl;
    std::cout << "  - 分布が非正規または裾の重い分布" << std::endl;
    std::cout << "  - 極端な値の影響を減らしたい" << std::endl;
    std::cout << "  - 探索的データ分析" << std::endl;

    std::cout << "\n=== サンプル実行完了 ===" << std::endl;

    return 0;
}
