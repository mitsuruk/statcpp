/**
 * @file missing_data.hpp
 * @brief 欠損値処理関数
 *
 * 欠損値の検出、パターン分析、多重代入、感度分析などの
 * 欠損データ処理機能を提供します。
 *
 * 主な機能:
 * - 欠損パターンの分析 (MCAR/MAR/MNAR分類)
 * - Little の MCAR 検定
 * - 多重代入 (PMM, ブートストラップEM)
 * - 感度分析 (パターン混合モデル, 選択モデル)
 * - 完全ケース分析
 */

#ifndef STATCPP_MISSING_DATA_HPP
#define STATCPP_MISSING_DATA_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "statcpp/basic_statistics.hpp"
#include "statcpp/correlation_covariance.hpp"
#include "statcpp/data_wrangling.hpp"
#include "statcpp/dispersion_spread.hpp"

namespace statcpp {

// ============================================================================
// Missing Data Pattern Classification (MCAR/MAR/MNAR)
// ============================================================================

/**
 * @brief 欠損メカニズムの種類
 *
 * 欠損データの発生メカニズムを分類するための列挙型です。
 * - MCAR: 欠損が完全にランダムに発生
 * - MAR: 欠損が観測された変数に依存
 * - MNAR: 欠損が欠損値自体に依存
 */
enum class missing_mechanism {
    mcar,       ///< Missing Completely At Random
    mar,        ///< Missing At Random
    mnar,       ///< Missing Not At Random
    unknown     ///< 判定不能
};

/**
 * @brief Little の MCAR 検定結果
 *
 * Little の MCAR 検定の結果を格納する構造体です。
 * χ²統計量、p値、自由度、および結果の解釈を含みます。
 */
struct mcar_test_result {
    double chi_square = 0.0;    ///< χ²統計量
    double p_value = 1.0;       ///< p値
    std::size_t df = 0;         ///< 自由度
    bool is_mcar = true;        ///< MCAR と判定されたか（p > 0.05）
    std::string interpretation; ///< 解釈
};

/**
 * @brief 欠損パターン情報
 *
 * データセット内の欠損パターンに関する情報を格納する構造体です。
 * 各パターンの出現頻度、変数ごとの欠損率、全体の欠損率などを含みます。
 */
struct missing_pattern_info {
    std::vector<std::vector<uint8_t>> patterns;    ///< 欠損パターン（1 = 欠損, 0 = 観測）
    std::vector<std::size_t> pattern_counts;       ///< 各パターンの出現数
    std::vector<double> missing_rates;             ///< 各変数の欠損率
    double overall_missing_rate = 0.0;             ///< 全体の欠損率
    std::size_t n_complete_cases = 0;              ///< 完全ケース数
    std::size_t n_patterns = 0;                    ///< 欠損パターン数
};

/**
 * @brief 欠損パターンを分析する
 *
 * データセット内の欠損パターンを分析し、各変数の欠損率、
 * 全体の欠損率、欠損パターンの種類と頻度を計算します。
 *
 * @param data 分析対象の2次元データ（行: 観測、列: 変数）
 * @return missing_pattern_info 欠損パターン情報
 * @throws std::invalid_argument データが空の場合、または行サイズが不一致の場合
 */
inline missing_pattern_info analyze_missing_patterns(
    const std::vector<std::vector<double>>& data)
{
    if (data.empty()) {
        throw std::invalid_argument(
            "statcpp::analyze_missing_patterns: empty data");
    }

    missing_pattern_info result;
    std::size_t n_rows = data.size();
    std::size_t n_cols = data[0].size();

    // 各変数の欠損率を計算
    result.missing_rates.resize(n_cols, 0.0);
    std::size_t total_missing = 0;

    for (const auto& row : data) {
        if (row.size() != n_cols) {
            throw std::invalid_argument(
                "statcpp::analyze_missing_patterns: inconsistent row sizes");
        }
        for (std::size_t j = 0; j < n_cols; ++j) {
            if (is_na(row[j])) {
                ++result.missing_rates[j];
                ++total_missing;
            }
        }
    }

    for (auto& rate : result.missing_rates) {
        rate /= static_cast<double>(n_rows);
    }
    result.overall_missing_rate = static_cast<double>(total_missing) /
                                  static_cast<double>(n_rows * n_cols);

    // 欠損パターンを抽出（std::mapを使わずに線形探索で実装）
    std::vector<std::vector<uint8_t>> unique_patterns;
    std::vector<std::size_t> pattern_counts;
    result.n_complete_cases = 0;

    for (const auto& row : data) {
        std::vector<uint8_t> pattern(n_cols);
        bool has_missing = false;
        for (std::size_t j = 0; j < n_cols; ++j) {
            pattern[j] = is_na(row[j]) ? 1 : 0;
            if (pattern[j]) {
                has_missing = true;
            }
        }
        if (!has_missing) {
            ++result.n_complete_cases;
        }

        // 既存のパターンを探す
        bool found = false;
        for (std::size_t i = 0; i < unique_patterns.size(); ++i) {
            if (unique_patterns[i] == pattern) {
                ++pattern_counts[i];
                found = true;
                break;
            }
        }

        // 新しいパターンなら追加
        if (!found) {
            unique_patterns.push_back(pattern);
            pattern_counts.push_back(1);
        }
    }

    // 結果を格納
    result.patterns = unique_patterns;
    result.pattern_counts = pattern_counts;
    result.n_patterns = result.patterns.size();

    return result;
}

/**
 * @brief 欠損値の指示変数を作成する
 *
 * データの各要素に対して、欠損値の場合は1.0、観測値の場合は0.0を
 * 持つ指示変数行列を作成します。
 *
 * @param data 入力データ（行: 観測、列: 変数）
 * @return std::vector<std::vector<double>> 欠損指示変数行列（1 = 欠損、0 = 観測）
 */
inline std::vector<std::vector<double>> create_missing_indicator(
    const std::vector<std::vector<double>>& data)
{
    std::vector<std::vector<double>> indicator;
    indicator.reserve(data.size());

    for (const auto& row : data) {
        std::vector<double> ind_row;
        ind_row.reserve(row.size());
        for (double val : row) {
            ind_row.push_back(is_na(val) ? 1.0 : 0.0);
        }
        indicator.push_back(std::move(ind_row));
    }
    return indicator;
}

/**
 * @brief Little の MCAR 検定（簡易版）
 *
 * 完全データと不完全データ間の平均差を検定し、
 * データが MCAR (Missing Completely At Random) かどうかを判定します。
 *
 * @note これは簡易版であり、完全な Little の検定には
 *       EMアルゴリズムによる共分散推定が必要です。
 *
 * @param data 検定対象の2次元データ（行: 観測、列: 変数）
 * @return mcar_test_result 検定結果（χ²統計量、p値、自由度、判定結果）
 * @throws std::invalid_argument データが空の場合
 */
inline mcar_test_result test_mcar_simple(
    const std::vector<std::vector<double>>& data)
{
    if (data.empty()) {
        throw std::invalid_argument("statcpp::test_mcar_simple: empty data");
    }

    std::size_t n_cols = data[0].size();
    mcar_test_result result;

    // 各変数について、他の変数の欠損有無と相関を調べる
    // MCAR なら、ある変数の欠損は他の変数の値と無関係
    double total_chi_sq = 0.0;
    std::size_t total_df = 0;

    for (std::size_t j = 0; j < n_cols; ++j) {
        // 変数 j の欠損指示変数
        std::vector<double> missing_j;
        missing_j.reserve(data.size());
        for (const auto& row : data) {
            missing_j.push_back(is_na(row[j]) ? 1.0 : 0.0);
        }

        // 他の変数との相関を調べる
        for (std::size_t k = 0; k < n_cols; ++k) {
            if (j == k) continue;

            // 変数 k の観測値（両方が観測されているケースのみ）
            std::vector<double> obs_k;
            std::vector<double> miss_j_subset;

            for (std::size_t i = 0; i < data.size(); ++i) {
                if (!is_na(data[i][k])) {
                    obs_k.push_back(data[i][k]);
                    miss_j_subset.push_back(missing_j[i]);
                }
            }

            if (obs_k.size() < 5) continue;  // サンプルサイズが小さすぎる

            // 欠損群と観測群の平均を比較
            std::vector<double> obs_when_j_missing;
            std::vector<double> obs_when_j_observed;

            for (std::size_t i = 0; i < obs_k.size(); ++i) {
                if (miss_j_subset[i] > 0.5) {
                    obs_when_j_missing.push_back(obs_k[i]);
                } else {
                    obs_when_j_observed.push_back(obs_k[i]);
                }
            }

            if (obs_when_j_missing.size() < 2 || obs_when_j_observed.size() < 2) {
                continue;
            }

            // 2標本t検定統計量の計算
            double mean1 = mean(obs_when_j_missing.begin(), obs_when_j_missing.end());
            double mean2 = mean(obs_when_j_observed.begin(), obs_when_j_observed.end());
            double var1 = var(obs_when_j_missing.begin(), obs_when_j_missing.end(), 1);
            double var2 = var(obs_when_j_observed.begin(), obs_when_j_observed.end(), 1);

            double n1 = static_cast<double>(obs_when_j_missing.size());
            double n2 = static_cast<double>(obs_when_j_observed.size());

            double se = std::sqrt(var1 / n1 + var2 / n2);
            if (se > 1e-10) {
                double t_stat = (mean1 - mean2) / se;
                total_chi_sq += t_stat * t_stat;
                ++total_df;
            }
        }
    }

    result.chi_square = total_chi_sq;
    result.df = total_df;

    // χ²分布からp値を近似（Wilson-Hilferty 近似）
    if (total_df > 0) {
        double z = std::pow(total_chi_sq / static_cast<double>(total_df), 1.0 / 3.0) -
                   (1.0 - 2.0 / (9.0 * static_cast<double>(total_df)));
        z /= std::sqrt(2.0 / (9.0 * static_cast<double>(total_df)));
        // 標準正規分布の上側確率（近似）
        result.p_value = 0.5 * std::erfc(z / std::sqrt(2.0));
        result.p_value = std::max(0.0, std::min(1.0, result.p_value));
    } else {
        result.p_value = 1.0;
    }

    result.is_mcar = (result.p_value > 0.05);
    result.interpretation = result.is_mcar
        ? "MCAR assumption is not rejected (p > 0.05). "
          "Missing data may be completely random."
        : "MCAR assumption is rejected (p <= 0.05). "
          "Missing data is likely MAR or MNAR.";

    return result;
}

/**
 * @brief 欠損メカニズムの簡易診断
 *
 * データの欠損メカニズム（MCAR, MAR, MNAR）を診断します。
 * 内部で Little の MCAR 検定を実行し、その結果に基づいて
 * 欠損メカニズムを推定します。
 *
 * @note MAR と MNAR の区別は観測データだけでは困難なため、
 *       MCAR が棄却された場合は保守的に MAR と判定します。
 *
 * @param data 診断対象の2次元データ（行: 観測、列: 変数）
 * @return missing_mechanism 推定された欠損メカニズム
 */
inline missing_mechanism diagnose_missing_mechanism(
    const std::vector<std::vector<double>>& data)
{
    auto mcar_result = test_mcar_simple(data);

    if (mcar_result.is_mcar) {
        return missing_mechanism::mcar;
    }

    // MAR vs MNAR の判定は観測データだけでは困難
    // ここでは MAR と仮定（保守的な選択）
    return missing_mechanism::mar;
}

// ============================================================================
// Multiple Imputation
// ============================================================================

/**
 * @brief 多重代入結果
 *
 * 多重代入法の結果を格納する構造体です。
 * 複数の代入データセット、プールされた統計量、
 * 代入内分散、代入間分散、欠損情報割合を含みます。
 */
struct multiple_imputation_result {
    std::vector<std::vector<std::vector<double>>> imputed_datasets;  ///< 代入されたデータセット群
    std::size_t m = 0;                                               ///< 代入数
    std::vector<double> pooled_means;                                ///< プールされた平均
    std::vector<double> pooled_vars;                                 ///< プールされた分散
    std::vector<double> within_vars;                                 ///< 代入内分散
    std::vector<double> between_vars;                                ///< 代入間分散
    std::vector<double> fraction_missing_info;                       ///< 欠損情報割合 (FMI)
};

/**
 * @brief 条件付き平均による単一代入
 *
 * 予測変数を用いた単純な線形回帰により、欠損値を条件付き平均で代入します。
 *
 * @param data 入力データ（行: 観測、列: 変数）
 * @param target_col 代入対象の列インデックス
 * @param predictor_cols 予測変数として使用する列のインデックス
 * @return std::vector<double> 代入後の target_col の値
 */
inline std::vector<double> impute_conditional_mean(
    const std::vector<std::vector<double>>& data,
    std::size_t target_col,
    const std::vector<std::size_t>& predictor_cols)
{
    std::size_t n = data.size();
    std::vector<double> result;
    result.reserve(n);

    // 完全ケースを抽出
    std::vector<std::vector<double>> complete_cases;
    for (const auto& row : data) {
        bool complete = !is_na(row[target_col]);
        for (std::size_t col : predictor_cols) {
            if (is_na(row[col])) {
                complete = false;
                break;
            }
        }
        if (complete) {
            complete_cases.push_back(row);
        }
    }

    if (complete_cases.empty()) {
        // 完全ケースがない場合は単純平均代入
        std::vector<double> non_na;
        for (const auto& row : data) {
            if (!is_na(row[target_col])) {
                non_na.push_back(row[target_col]);
            }
        }
        double fill_val = non_na.empty() ? 0.0 : mean(non_na.begin(), non_na.end());
        for (const auto& row : data) {
            result.push_back(is_na(row[target_col]) ? fill_val : row[target_col]);
        }
        return result;
    }

    // 単純な線形回帰で代入値を推定
    // Y = target_col, X = predictor_cols (最初の予測変数のみ使用する簡易版)
    if (predictor_cols.empty()) {
        double m = 0.0;
        for (const auto& row : complete_cases) {
            m += row[target_col];
        }
        m /= static_cast<double>(complete_cases.size());
        for (const auto& row : data) {
            result.push_back(is_na(row[target_col]) ? m : row[target_col]);
        }
        return result;
    }

    // 最初の予測変数を使った単回帰
    std::size_t pred_col = predictor_cols[0];
    std::vector<double> x_vals, y_vals;
    for (const auto& row : complete_cases) {
        x_vals.push_back(row[pred_col]);
        y_vals.push_back(row[target_col]);
    }

    double x_mean = mean(x_vals.begin(), x_vals.end());
    double y_mean = mean(y_vals.begin(), y_vals.end());

    double cov_xy = 0.0;
    double var_x = 0.0;
    for (std::size_t i = 0; i < x_vals.size(); ++i) {
        double dx = x_vals[i] - x_mean;
        double dy = y_vals[i] - y_mean;
        cov_xy += dx * dy;
        var_x += dx * dx;
    }

    double beta = (var_x > 1e-10) ? cov_xy / var_x : 0.0;
    double alpha = y_mean - beta * x_mean;

    // 代入
    for (const auto& row : data) {
        if (is_na(row[target_col])) {
            if (!is_na(row[pred_col])) {
                result.push_back(alpha + beta * row[pred_col]);
            } else {
                result.push_back(y_mean);
            }
        } else {
            result.push_back(row[target_col]);
        }
    }

    return result;
}

/**
 * @brief 多重代入（PMM: Predictive Mean Matching）
 *
 * Predictive Mean Matching 法による多重代入を実行します。
 * 予測値に最も近い k 個の観測値からランダムにドナーを選択して代入します。
 * 結果は Rubin のルールに基づいてプールされます。
 *
 * @param data 入力データ（行: 観測、列: 変数）
 * @param m 代入回数（デフォルト: 5）
 * @param seed 乱数シード（0の場合はランダムシード）
 * @return multiple_imputation_result 多重代入結果
 * @throws std::invalid_argument データが空の場合
 */
inline multiple_imputation_result multiple_imputation_pmm(
    const std::vector<std::vector<double>>& data,
    std::size_t m = 5,
    unsigned int seed = 0)
{
    if (data.empty()) {
        throw std::invalid_argument(
            "statcpp::multiple_imputation_pmm: empty data");
    }

    std::size_t n_rows = data.size();
    std::size_t n_cols = data[0].size();

    multiple_imputation_result result;
    result.m = m;
    result.imputed_datasets.resize(m);

    // 乱数生成器
    std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);

    for (std::size_t imp = 0; imp < m; ++imp) {
        // データをコピー
        auto imputed = data;

        // 各列について代入
        for (std::size_t j = 0; j < n_cols; ++j) {
            // 欠損しているインデックスを収集
            std::vector<std::size_t> missing_indices;
            std::vector<std::size_t> observed_indices;
            std::vector<double> observed_values;

            for (std::size_t i = 0; i < n_rows; ++i) {
                if (is_na(imputed[i][j])) {
                    missing_indices.push_back(i);
                } else {
                    observed_indices.push_back(i);
                    observed_values.push_back(imputed[i][j]);
                }
            }

            if (missing_indices.empty() || observed_values.empty()) {
                continue;
            }

            // 予測変数として他の列を使用
            std::vector<std::size_t> predictor_cols;
            for (std::size_t k = 0; k < n_cols; ++k) {
                if (k != j) {
                    predictor_cols.push_back(k);
                }
            }

            // 条件付き平均を計算
            auto cond_means = impute_conditional_mean(imputed, j, predictor_cols);

            // PMM: 予測値に最も近い k 個の観測値からランダムに選択
            constexpr std::size_t k_donors = 5;

            for (std::size_t idx : missing_indices) {
                double pred_val = cond_means[idx];

                // 観測値との距離を計算
                std::vector<std::pair<double, std::size_t>> distances;
                distances.reserve(observed_indices.size());
                for (std::size_t oi = 0; oi < observed_indices.size(); ++oi) {
                    std::size_t obs_idx = observed_indices[oi];
                    double dist = std::abs(observed_values[oi] - pred_val);
                    distances.emplace_back(dist, obs_idx);
                }

                // 距離でソート
                std::partial_sort(distances.begin(),
                                  distances.begin() + std::min(k_donors, distances.size()),
                                  distances.end());

                // k個のドナーからランダムに選択
                std::size_t n_donors = std::min(k_donors, distances.size());
                std::uniform_int_distribution<std::size_t> dist(0, n_donors - 1);
                std::size_t donor_idx = distances[dist(rng)].second;

                imputed[idx][j] = imputed[donor_idx][j];
            }
        }

        result.imputed_datasets[imp] = std::move(imputed);
    }

    // Rubin のルールによるプーリング
    result.pooled_means.resize(n_cols, 0.0);
    result.pooled_vars.resize(n_cols, 0.0);
    result.within_vars.resize(n_cols, 0.0);
    result.between_vars.resize(n_cols, 0.0);
    result.fraction_missing_info.resize(n_cols, 0.0);

    for (std::size_t j = 0; j < n_cols; ++j) {
        std::vector<double> means_j(m);
        std::vector<double> vars_j(m);

        for (std::size_t imp = 0; imp < m; ++imp) {
            std::vector<double> col_values;
            col_values.reserve(n_rows);
            for (std::size_t i = 0; i < n_rows; ++i) {
                col_values.push_back(result.imputed_datasets[imp][i][j]);
            }
            means_j[imp] = mean(col_values.begin(), col_values.end());
            vars_j[imp] = var(col_values.begin(), col_values.end(), 1);
        }

        // プールされた平均
        result.pooled_means[j] = mean(means_j.begin(), means_j.end());

        // 代入内分散 (W)
        result.within_vars[j] = mean(vars_j.begin(), vars_j.end());

        // 代入間分散 (B)
        double b = 0.0;
        for (double mj : means_j) {
            double diff = mj - result.pooled_means[j];
            b += diff * diff;
        }
        result.between_vars[j] = b / static_cast<double>(m - 1);

        // 全分散
        result.pooled_vars[j] = result.within_vars[j] +
                                (1.0 + 1.0 / static_cast<double>(m)) * result.between_vars[j];

        // 欠損情報割合 (FMI)
        // Rubin (1987) の定義: FMI = (r + 2/(df+3)) / (r+1)
        // ここで r = (1 + 1/m) * B / W
        // 簡易版として FMI ≈ (1 + 1/m) * B / T を使用
        if (result.pooled_vars[j] > 1e-10) {
            result.fraction_missing_info[j] =
                (1.0 + 1.0 / static_cast<double>(m)) * result.between_vars[j] /
                result.pooled_vars[j];
        }
    }

    return result;
}

/**
 * @brief 多重代入（ブートストラップEM法の簡易版）
 *
 * ブートストラップサンプリングと正規分布からの確率的代入を組み合わせた
 * 多重代入法を実行します。結果は Rubin のルールに基づいてプールされます。
 *
 * @param data 入力データ（行: 観測、列: 変数）
 * @param m 代入回数（デフォルト: 5）
 * @param seed 乱数シード（0の場合はランダムシード）
 * @return multiple_imputation_result 多重代入結果
 * @throws std::invalid_argument データが空の場合
 */
inline multiple_imputation_result multiple_imputation_bootstrap(
    const std::vector<std::vector<double>>& data,
    std::size_t m = 5,
    unsigned int seed = 0)
{
    if (data.empty()) {
        throw std::invalid_argument(
            "statcpp::multiple_imputation_bootstrap: empty data");
    }

    std::size_t n_rows = data.size();
    std::size_t n_cols = data[0].size();

    multiple_imputation_result result;
    result.m = m;
    result.imputed_datasets.resize(m);

    std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);

    for (std::size_t imp = 0; imp < m; ++imp) {
        // ブートストラップサンプルを作成（観測値から）
        std::vector<std::vector<double>> boot_sample;
        boot_sample.reserve(n_rows);

        std::uniform_int_distribution<std::size_t> dist(0, n_rows - 1);
        for (std::size_t i = 0; i < n_rows; ++i) {
            boot_sample.push_back(data[dist(rng)]);
        }

        // 各列の統計量を計算
        std::vector<double> col_means(n_cols);
        std::vector<double> col_stds(n_cols);

        for (std::size_t j = 0; j < n_cols; ++j) {
            std::vector<double> non_na;
            for (const auto& row : boot_sample) {
                if (!is_na(row[j])) {
                    non_na.push_back(row[j]);
                }
            }
            if (!non_na.empty()) {
                col_means[j] = mean(non_na.begin(), non_na.end());
                col_stds[j] = non_na.size() > 1 ?
                    std::sqrt(var(non_na.begin(), non_na.end(), 1)) : 0.0;
            }
        }

        // 代入（正規分布からの確率的代入）
        auto imputed = data;
        std::normal_distribution<double> normal(0.0, 1.0);

        for (std::size_t i = 0; i < n_rows; ++i) {
            for (std::size_t j = 0; j < n_cols; ++j) {
                if (is_na(imputed[i][j])) {
                    // 平均 + ランダムノイズ
                    imputed[i][j] = col_means[j] + col_stds[j] * normal(rng);
                }
            }
        }

        result.imputed_datasets[imp] = std::move(imputed);
    }

    // Rubin のルールによるプーリング
    result.pooled_means.resize(n_cols, 0.0);
    result.pooled_vars.resize(n_cols, 0.0);
    result.within_vars.resize(n_cols, 0.0);
    result.between_vars.resize(n_cols, 0.0);
    result.fraction_missing_info.resize(n_cols, 0.0);

    for (std::size_t j = 0; j < n_cols; ++j) {
        std::vector<double> means_j(m);
        std::vector<double> vars_j(m);

        for (std::size_t imp = 0; imp < m; ++imp) {
            std::vector<double> col_values;
            col_values.reserve(n_rows);
            for (std::size_t i = 0; i < n_rows; ++i) {
                col_values.push_back(result.imputed_datasets[imp][i][j]);
            }
            means_j[imp] = mean(col_values.begin(), col_values.end());
            vars_j[imp] = var(col_values.begin(), col_values.end(), 1);
        }

        result.pooled_means[j] = mean(means_j.begin(), means_j.end());
        result.within_vars[j] = mean(vars_j.begin(), vars_j.end());

        double b = 0.0;
        for (double mj : means_j) {
            double diff = mj - result.pooled_means[j];
            b += diff * diff;
        }
        result.between_vars[j] = b / static_cast<double>(m - 1);

        result.pooled_vars[j] = result.within_vars[j] +
                                (1.0 + 1.0 / static_cast<double>(m)) * result.between_vars[j];

        // 欠損情報割合 (FMI): Rubin (1987)
        if (result.pooled_vars[j] > 1e-10) {
            result.fraction_missing_info[j] =
                (1.0 + 1.0 / static_cast<double>(m)) * result.between_vars[j] /
                result.pooled_vars[j];
        }
    }

    return result;
}

// ============================================================================
// Sensitivity Analysis for Missing Data
// ============================================================================

/**
 * @brief 感度分析結果（単一パラメータ）
 *
 * 欠損データに対する感度分析の結果を格納する構造体です。
 * 感度パラメータの各値に対する推定平均・分散、および結果の解釈を含みます。
 */
struct sensitivity_analysis_result {
    std::vector<double> delta_values;          ///< 感度パラメータの値
    std::vector<double> estimated_means;       ///< 推定された平均
    std::vector<double> estimated_vars;        ///< 推定された分散
    double original_mean = 0.0;                ///< 元の推定平均
    double original_var = 0.0;                 ///< 元の推定分散
    std::string interpretation;                ///< 解釈
};

/**
 * @brief パターン混合モデルによる感度分析
 *
 * MNAR (Missing Not At Random) の仮定下で、欠損値と観測値の間の
 * 差（delta）を変化させて推定値の頑健性を評価します。
 *
 * パターン混合モデルの基本式:
 * - E[Y] = E[Y|R=1] * P(R=1) + E[Y|R=0] * P(R=0)
 * - ここで R=1 は観測、R=0 は欠損を示す
 * - 感度パラメータ delta: E[Y|R=0] = E[Y|R=1] + delta
 *
 * delta の解釈:
 * - delta = 0: MAR（欠損は観測値と同じ分布に従う）
 * - delta > 0: 欠損値は観測値より高い傾向
 * - delta < 0: 欠損値は観測値より低い傾向
 *
 * @note この実装は分散について簡易的な仮定（観測分散と同等）を使用しています。
 *       より厳密な分析には、欠損パターン別の分散推定が必要です。
 *
 * @param data 分析対象の1次元データ
 * @param delta_values 感度パラメータ delta の値のベクトル
 * @return sensitivity_analysis_result 感度分析結果
 * @throws std::invalid_argument データが空の場合、または全て欠損の場合
 */
inline sensitivity_analysis_result sensitivity_analysis_pattern_mixture(
    const std::vector<double>& data,
    const std::vector<double>& delta_values)
{
    if (data.empty()) {
        throw std::invalid_argument(
            "statcpp::sensitivity_analysis_pattern_mixture: empty data");
    }

    sensitivity_analysis_result result;
    result.delta_values = delta_values;
    result.estimated_means.reserve(delta_values.size());
    result.estimated_vars.reserve(delta_values.size());

    // 観測値の統計量
    std::vector<double> observed;
    std::size_t n_missing = 0;
    for (double val : data) {
        if (!is_na(val)) {
            observed.push_back(val);
        } else {
            ++n_missing;
        }
    }

    if (observed.empty()) {
        throw std::invalid_argument(
            "statcpp::sensitivity_analysis_pattern_mixture: all values are missing");
    }

    double obs_mean = mean(observed.begin(), observed.end());
    double obs_var = observed.size() > 1 ?
        var(observed.begin(), observed.end(), 1) : 0.0;

    result.original_mean = obs_mean;
    result.original_var = obs_var;

    double n_obs = static_cast<double>(observed.size());
    double n_total = static_cast<double>(data.size());
    double prop_obs = n_obs / n_total;
    double prop_miss = static_cast<double>(n_missing) / n_total;

    // 各 delta 値について推定
    for (double delta : delta_values) {
        // パターン混合モデル: E[Y] = E[Y|R=1] * P(R=1) + E[Y|R=0] * P(R=0)
        // E[Y|R=0] = E[Y|R=1] + delta
        double imputed_mean = obs_mean + delta;
        double overall_mean = obs_mean * prop_obs + imputed_mean * prop_miss;

        // 分散の推定（簡易版）
        // 真の分散は追加の仮定が必要だが、ここでは観測分散を使用
        double overall_var = obs_var;

        result.estimated_means.push_back(overall_mean);
        result.estimated_vars.push_back(overall_var);
    }

    result.interpretation =
        "Pattern mixture model sensitivity analysis. "
        "delta represents the hypothesized difference between "
        "missing and observed values. delta=0 corresponds to MAR assumption.";

    return result;
}

/**
 * @brief 選択モデルによる感度分析
 *
 * 欠損が応答値に依存する程度（phi）を変化させて、推定値の頑健性を評価します。
 * phi = 0 は MAR の仮定に対応し、phi > 0 は欠損値が観測値より
 * 低い傾向にあることを示します。
 *
 * @param data 分析対象の1次元データ
 * @param phi_values 感度パラメータ phi の値のベクトル
 * @return sensitivity_analysis_result 感度分析結果
 * @throws std::invalid_argument データが空の場合、または全て欠損の場合
 */
inline sensitivity_analysis_result sensitivity_analysis_selection_model(
    const std::vector<double>& data,
    const std::vector<double>& phi_values)
{
    if (data.empty()) {
        throw std::invalid_argument(
            "statcpp::sensitivity_analysis_selection_model: empty data");
    }

    sensitivity_analysis_result result;
    result.delta_values = phi_values;  // phi を delta として格納
    result.estimated_means.reserve(phi_values.size());
    result.estimated_vars.reserve(phi_values.size());

    // 観測値の統計量
    std::vector<double> observed;
    std::size_t n_missing = 0;
    for (double val : data) {
        if (!is_na(val)) {
            observed.push_back(val);
        } else {
            ++n_missing;
        }
    }

    if (observed.empty()) {
        throw std::invalid_argument(
            "statcpp::sensitivity_analysis_selection_model: all values are missing");
    }

    double obs_mean = mean(observed.begin(), observed.end());
    double obs_var = observed.size() > 1 ?
        var(observed.begin(), observed.end(), 1) : 0.0;
    double obs_std = std::sqrt(obs_var);

    result.original_mean = obs_mean;
    result.original_var = obs_var;

    // 選択モデル: logit(P(R=1|Y)) = alpha + phi * Y
    // phi > 0: 高い値ほど観測されやすい（欠損値は低い傾向）
    // phi < 0: 低い値ほど観測されやすい（欠損値は高い傾向）
    // phi = 0: MAR

    for (double phi : phi_values) {
        // 簡易的な補正: 欠損値の期待値を phi に基づいて調整
        // phi が正なら欠損値は観測値より低い傾向
        double adjustment = -phi * obs_std * 0.5;  // スケーリング係数
        double imputed_mean = obs_mean + adjustment;

        double n_obs = static_cast<double>(observed.size());
        double n_total = static_cast<double>(data.size());
        double prop_obs = n_obs / n_total;
        double prop_miss = static_cast<double>(n_missing) / n_total;

        double overall_mean = obs_mean * prop_obs + imputed_mean * prop_miss;
        double overall_var = obs_var;

        result.estimated_means.push_back(overall_mean);
        result.estimated_vars.push_back(overall_var);
    }

    result.interpretation =
        "Selection model sensitivity analysis. "
        "phi represents the dependence of missingness on the outcome value. "
        "phi=0 corresponds to MAR assumption. "
        "phi>0 implies missing values tend to be lower than observed values.";

    return result;
}

/**
 * @brief ティッピングポイント分析結果
 *
 * ティッピングポイント分析の結果を格納する構造体です。
 * 結論が変わる臨界点（ティッピングポイント）の情報を含みます。
 */
struct tipping_point_result {
    double tipping_point = 0.0;        ///< ティッピングポイント（臨界 delta 値）
    bool found = false;                ///< ティッピングポイントが見つかったか
    double threshold = 0.0;            ///< 使用した閾値
    std::string interpretation;        ///< 解釈
};

/**
 * @brief ティッピングポイント分析
 *
 * 推定平均が指定した閾値を横切る臨界点（ティッピングポイント）を探索します。
 * これにより、MNAR の仮定がどの程度極端になれば結論が変わるかを評価できます。
 *
 * @param data 分析対象の1次元データ
 * @param threshold 閾値（例: 帰無仮説の値）
 * @param delta_min 探索する delta の最小値（デフォルト: -5.0）
 * @param delta_max 探索する delta の最大値（デフォルト: 5.0）
 * @param n_points 探索点数（デフォルト: 100）
 * @return tipping_point_result ティッピングポイント分析結果
 */
inline tipping_point_result find_tipping_point(
    const std::vector<double>& data,
    double threshold = 0.0,            // 例: 帰無仮説の値
    double delta_min = -5.0,
    double delta_max = 5.0,
    std::size_t n_points = 100)
{
    tipping_point_result result;
    result.threshold = threshold;
    result.found = false;
    result.tipping_point = NA;

    std::vector<double> delta_values;
    double step = (delta_max - delta_min) / static_cast<double>(n_points - 1);
    for (std::size_t i = 0; i < n_points; ++i) {
        delta_values.push_back(delta_min + static_cast<double>(i) * step);
    }

    auto sens_result = sensitivity_analysis_pattern_mixture(data, delta_values);

    // 閾値を横切る点を探す
    for (std::size_t i = 1; i < sens_result.estimated_means.size(); ++i) {
        double prev = sens_result.estimated_means[i - 1];
        double curr = sens_result.estimated_means[i];

        if ((prev <= threshold && curr > threshold) ||
            (prev >= threshold && curr < threshold)) {
            // 線形補間でティッピングポイントを推定
            double delta_prev = sens_result.delta_values[i - 1];
            double delta_curr = sens_result.delta_values[i];
            double ratio = (threshold - prev) / (curr - prev);
            result.tipping_point = delta_prev + ratio * (delta_curr - delta_prev);
            result.found = true;
            break;
        }
    }

    if (result.found) {
        result.interpretation =
            "Tipping point found at delta = " + std::to_string(result.tipping_point) +
            ". At this value of delta (shift in missing values), "
            "the estimated mean crosses the threshold of " +
            std::to_string(threshold) + ".";
    } else {
        result.interpretation =
            "No tipping point found in the specified range. "
            "The conclusion is robust to MNAR assumptions within this range.";
    }

    return result;
}

// ============================================================================
// Complete Case Analysis Utilities
// ============================================================================

/**
 * @brief 完全ケース分析の結果
 *
 * 完全ケース分析（listwise deletion）の結果を格納する構造体です。
 * 完全ケースのデータ、完全ケース数、削除されたケース数を含みます。
 */
struct complete_case_result {
    std::vector<std::vector<double>> complete_data;    ///< 完全ケースのみのデータ
    std::size_t n_complete = 0;                        ///< 完全ケース数
    std::size_t n_dropped = 0;                         ///< 削除されたケース数
    double proportion_complete = 0.0;                  ///< 完全ケースの割合
};

/**
 * @brief 完全ケースを抽出する
 *
 * データセットから欠損値を含まない行（完全ケース）のみを抽出します。
 * これは listwise deletion とも呼ばれます。
 *
 * @param data 入力データ（行: 観測、列: 変数）
 * @return complete_case_result 完全ケース分析結果
 */
inline complete_case_result extract_complete_cases(
    const std::vector<std::vector<double>>& data)
{
    complete_case_result result;
    result.n_dropped = 0;

    for (const auto& row : data) {
        bool is_complete = true;
        for (double val : row) {
            if (is_na(val)) {
                is_complete = false;
                break;
            }
        }
        if (is_complete) {
            result.complete_data.push_back(row);
        } else {
            ++result.n_dropped;
        }
    }

    result.n_complete = result.complete_data.size();
    result.proportion_complete = data.empty() ? 0.0 :
        static_cast<double>(result.n_complete) / static_cast<double>(data.size());

    return result;
}

/**
 * @brief 利用可能ケース分析による相関行列（pairwise deletion）
 *
 * 各変数ペアについて、両方が観測されているケースのみを使用して
 * 相関係数を計算します。これは pairwise deletion とも呼ばれます。
 *
 * @param data 入力データ（行: 観測、列: 変数）
 * @return std::vector<std::vector<double>> 相関行列（観測数が不足の場合は NA）
 */
inline std::vector<std::vector<double>> correlation_matrix_pairwise(
    const std::vector<std::vector<double>>& data)
{
    if (data.empty()) {
        return {};
    }

    std::size_t n_cols = data[0].size();
    std::vector<std::vector<double>> corr_matrix(n_cols, std::vector<double>(n_cols, 1.0));

    for (std::size_t i = 0; i < n_cols; ++i) {
        for (std::size_t j = i + 1; j < n_cols; ++j) {
            // 両方が観測されているケースを抽出
            std::vector<double> x_vals, y_vals;
            for (const auto& row : data) {
                if (!is_na(row[i]) && !is_na(row[j])) {
                    x_vals.push_back(row[i]);
                    y_vals.push_back(row[j]);
                }
            }

            if (x_vals.size() >= 2) {
                double r = pearson_correlation(x_vals.begin(), x_vals.end(),
                                               y_vals.begin(), y_vals.end());
                corr_matrix[i][j] = r;
                corr_matrix[j][i] = r;
            } else {
                corr_matrix[i][j] = NA;
                corr_matrix[j][i] = NA;
            }
        }
    }

    return corr_matrix;
}

}  // namespace statcpp

#endif  // STATCPP_MISSING_DATA_HPP
