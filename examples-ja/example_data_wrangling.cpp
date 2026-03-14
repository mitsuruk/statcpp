/**
 * @file example_data_wrangling.cpp
 * @brief データラングリングのサンプルコード
 *
 * 欠損値処理、データ変換、フィルタリング、集約等の
 * データ前処理機能の使用例を示します。
 */

#include <iostream>
#include <iomanip>
#include <string>
#include "statcpp/data_wrangling.hpp"

int main() {
    std::cout << "=== データラングリングの例 ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    // ============================================================================
    // 1. 欠損値の処理
    // ============================================================================
    std::cout << "\n1. 欠損値の処理" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> data_with_na = {1.0, 2.0, statcpp::NA, 4.0, 5.0, statcpp::NA, 7.0};

    std::cout << "元のデータ: [";
    for (std::size_t i = 0; i < data_with_na.size(); ++i) {
        if (statcpp::is_na(data_with_na[i])) {
            std::cout << "NA";
        } else {
            std::cout << data_with_na[i];
        }
        if (i + 1 < data_with_na.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // NAを削除
    auto dropped = statcpp::dropna(data_with_na);
    std::cout << "\nNA削除後: [";
    for (std::size_t i = 0; i < dropped.size(); ++i) {
        std::cout << dropped[i];
        if (i + 1 < dropped.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // NAを0で埋める
    auto filled_zero = statcpp::fillna(data_with_na, 0.0);
    std::cout << "0で埋める: [";
    for (std::size_t i = 0; i < filled_zero.size(); ++i) {
        std::cout << filled_zero[i];
        if (i + 1 < filled_zero.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // NAを平均値で埋める
    auto filled_mean = statcpp::fillna_mean(data_with_na);
    std::cout << "平均値で埋める: [";
    for (std::size_t i = 0; i < filled_mean.size(); ++i) {
        std::cout << filled_mean[i];
        if (i + 1 < filled_mean.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // 線形補間
    auto interpolated = statcpp::fillna_interpolate(data_with_na);
    std::cout << "線形補間: [";
    for (std::size_t i = 0; i < interpolated.size(); ++i) {
        std::cout << interpolated[i];
        if (i + 1 < interpolated.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // ============================================================================
    // 2. フィルタリング
    // ============================================================================
    std::cout << "\n2. フィルタリング" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // 偶数のみフィルタ
    auto evens = statcpp::filter(numbers, [](double x) { return static_cast<int>(x) % 2 == 0; });
    std::cout << "偶数: [";
    for (std::size_t i = 0; i < evens.size(); ++i) {
        std::cout << evens[i];
        if (i + 1 < evens.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // 範囲でフィルタ（3以上7以下）
    auto range_filtered = statcpp::filter_range(numbers, 3.0, 7.0);
    std::cout << "範囲 [3, 7]: [";
    for (std::size_t i = 0; i < range_filtered.size(); ++i) {
        std::cout << range_filtered[i];
        if (i + 1 < range_filtered.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // ============================================================================
    // 3. 変換
    // ============================================================================
    std::cout << "\n3. 変換" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> transform_data = {1.0, 4.0, 9.0, 16.0, 25.0};

    // 平方根変換
    auto sqrt_data = statcpp::sqrt_transform(transform_data);
    std::cout << "平方根変換: [";
    for (std::size_t i = 0; i < sqrt_data.size(); ++i) {
        std::cout << sqrt_data[i];
        if (i + 1 < sqrt_data.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // 対数変換
    auto log_data = statcpp::log_transform(transform_data);
    std::cout << "対数変換: [";
    for (std::size_t i = 0; i < log_data.size(); ++i) {
        std::cout << log_data[i];
        if (i + 1 < log_data.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Box-Cox変換（lambda = 0.5）
    auto boxcox_data = statcpp::boxcox_transform(transform_data, 0.5);
    std::cout << "Box-Cox変換 (lambda=0.5): [";
    for (std::size_t i = 0; i < boxcox_data.size(); ++i) {
        std::cout << boxcox_data[i];
        if (i + 1 < boxcox_data.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // 順位変換
    std::vector<double> rank_data = {3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0};
    auto ranks = statcpp::rank_transform(rank_data);
    std::cout << "\n順位変換:" << std::endl;
    std::cout << "元のデータ: [";
    for (std::size_t i = 0; i < rank_data.size(); ++i) {
        std::cout << rank_data[i];
        if (i + 1 < rank_data.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "順位:       [";
    for (std::size_t i = 0; i < ranks.size(); ++i) {
        std::cout << ranks[i];
        if (i + 1 < ranks.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // ============================================================================
    // 4. グループ化と集約
    // ============================================================================
    std::cout << "\n4. グループ化と集約" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<std::string> groups = {"A", "B", "A", "B", "A", "B", "C", "C"};
    std::vector<double> values = {10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0};

    auto mean_result = statcpp::group_mean(groups, values);
    std::cout << "グループ別平均:" << std::endl;
    for (std::size_t i = 0; i < mean_result.keys.size(); ++i) {
        std::cout << "  グループ " << mean_result.keys[i] << ": " << mean_result.values[i] << std::endl;
    }

    auto sum_result = statcpp::group_sum(groups, values);
    std::cout << "\nグループ別合計:" << std::endl;
    for (std::size_t i = 0; i < sum_result.keys.size(); ++i) {
        std::cout << "  グループ " << sum_result.keys[i] << ": " << sum_result.values[i] << std::endl;
    }

    auto count_result = statcpp::group_count(groups, values);
    std::cout << "\nグループ別カウント:" << std::endl;
    for (std::size_t i = 0; i < count_result.keys.size(); ++i) {
        std::cout << "  グループ " << count_result.keys[i] << ": " << count_result.values[i] << std::endl;
    }

    // ============================================================================
    // 5. ソート
    // ============================================================================
    std::cout << "\n5. ソート" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> unsorted = {3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0};

    auto sorted_asc = statcpp::sort_values(unsorted, true);
    std::cout << "昇順ソート: [";
    for (std::size_t i = 0; i < sorted_asc.size(); ++i) {
        std::cout << sorted_asc[i];
        if (i + 1 < sorted_asc.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    auto indices = statcpp::argsort(unsorted, true);
    std::cout << "ソートインデックス: [";
    for (std::size_t i = 0; i < indices.size(); ++i) {
        std::cout << indices[i];
        if (i + 1 < indices.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // ============================================================================
    // 6. サンプリング
    // ============================================================================
    std::cout << "\n6. サンプリング" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    statcpp::set_seed(42);

    std::vector<int> population = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // 復元抽出
    auto sample_with = statcpp::sample_with_replacement(population, 5);
    std::cout << "復元抽出 (n=5): [";
    for (std::size_t i = 0; i < sample_with.size(); ++i) {
        std::cout << sample_with[i];
        if (i + 1 < sample_with.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // 非復元抽出
    auto sample_without = statcpp::sample_without_replacement(population, 5);
    std::cout << "非復元抽出 (n=5): [";
    for (std::size_t i = 0; i < sample_without.size(); ++i) {
        std::cout << sample_without[i];
        if (i + 1 < sample_without.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // 層化サンプリング
    std::vector<std::string> strata = {"A", "A", "A", "A", "B", "B", "B", "C", "C", "C"};
    std::vector<int> strata_values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto stratified = statcpp::stratified_sample(strata, strata_values, 0.5);
    std::cout << "層化サンプリング (50%): [";
    for (std::size_t i = 0; i < stratified.size(); ++i) {
        std::cout << stratified[i];
        if (i + 1 < stratified.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // ============================================================================
    // 7. 重複の処理
    // ============================================================================
    std::cout << "\n7. 重複の処理" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<int> duplicates = {1, 2, 2, 3, 3, 3, 4, 5, 5};

    auto unique = statcpp::drop_duplicates(duplicates);
    std::cout << "重複削除: [";
    for (std::size_t i = 0; i < unique.size(); ++i) {
        std::cout << unique[i];
        if (i + 1 < unique.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    auto counts = statcpp::value_counts(duplicates);
    std::cout << "\n値の出現回数:" << std::endl;
    for (const auto& pair : counts) {
        std::cout << "  " << pair.first << ": " << pair.second << std::endl;
    }

    auto dup_values = statcpp::get_duplicates(duplicates);
    std::cout << "\n重複している値: [";
    for (std::size_t i = 0; i < dup_values.size(); ++i) {
        std::cout << dup_values[i];
        if (i + 1 < dup_values.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // ============================================================================
    // 8. 移動統計量
    // ============================================================================
    std::cout << "\n8. 移動統計量" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> time_series = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    auto rolling_mean_3 = statcpp::rolling_mean(time_series, 3);
    std::cout << "移動平均 (ウィンドウ=3): [";
    for (std::size_t i = 0; i < rolling_mean_3.size(); ++i) {
        std::cout << rolling_mean_3[i];
        if (i + 1 < rolling_mean_3.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    auto rolling_std_3 = statcpp::rolling_std(time_series, 3);
    std::cout << "移動標準偏差 (ウィンドウ=3): [";
    for (std::size_t i = 0; i < rolling_std_3.size(); ++i) {
        std::cout << rolling_std_3[i];
        if (i + 1 < rolling_std_3.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    auto rolling_sum_3 = statcpp::rolling_sum(time_series, 3);
    std::cout << "移動合計 (ウィンドウ=3): [";
    for (std::size_t i = 0; i < rolling_sum_3.size(); ++i) {
        std::cout << rolling_sum_3[i];
        if (i + 1 < rolling_sum_3.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // ============================================================================
    // 9. カテゴリカルエンコーディング
    // ============================================================================
    std::cout << "\n9. カテゴリカルエンコーディング" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<std::string> categories = {"red", "blue", "green", "red", "green", "blue"};

    auto encoded = statcpp::label_encode(categories);
    std::cout << "ラベルエンコーディング:" << std::endl;
    std::cout << "  エンコード済み: [";
    for (std::size_t i = 0; i < encoded.encoded.size(); ++i) {
        std::cout << encoded.encoded[i];
        if (i + 1 < encoded.encoded.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  クラス: [";
    for (std::size_t i = 0; i < encoded.classes.size(); ++i) {
        std::cout << encoded.classes[i];
        if (i + 1 < encoded.classes.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // ビニング（等幅）
    std::vector<double> continuous = {1.5, 2.3, 4.7, 5.8, 7.2, 9.1, 10.5};
    auto bins_width = statcpp::bin_equal_width(continuous, 3);
    std::cout << "\nビニング (等幅, n=3):" << std::endl;
    std::cout << "  元のデータ: [";
    for (std::size_t i = 0; i < continuous.size(); ++i) {
        std::cout << continuous[i];
        if (i + 1 < continuous.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  ビン:       [";
    for (std::size_t i = 0; i < bins_width.size(); ++i) {
        std::cout << bins_width[i];
        if (i + 1 < bins_width.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // ============================================================================
    // 10. データ検証
    // ============================================================================
    std::cout << "\n10. データ検証" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> test_data = {1.0, 2.0, statcpp::NA, 4.0, -1.0,
                                      std::numeric_limits<double>::infinity()};

    auto validation = statcpp::validate_data(test_data, true, true, true);
    std::cout << "検証 (すべて許可):" << std::endl;
    std::cout << "  有効: " << (validation.is_valid ? "はい" : "いいえ") << std::endl;
    std::cout << "  欠損値: " << validation.n_missing << std::endl;
    std::cout << "  無限値: " << validation.n_infinite << std::endl;
    std::cout << "  負の値: " << validation.n_negative << std::endl;

    auto strict_validation = statcpp::validate_data(test_data, false, false, false);
    std::cout << "\n検証 (厳格):" << std::endl;
    std::cout << "  有効: " << (strict_validation.is_valid ? "はい" : "いいえ") << std::endl;

    std::vector<double> range_data = {1.0, 2.0, 3.0, 4.0, 5.0};
    bool in_range = statcpp::validate_range(range_data, 0.0, 10.0);
    std::cout << "\n範囲検証 [0, 10]: " << (in_range ? "合格" : "不合格") << std::endl;

    std::cout << "\n=== 例が正常に完了しました ===" << std::endl;

    return 0;
}
