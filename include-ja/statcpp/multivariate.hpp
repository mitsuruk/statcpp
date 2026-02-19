/**
 * @file multivariate.hpp
 * @brief 多変量解析の関数
 *
 * 共分散行列、相関行列、主成分分析（PCA）、データの標準化など、
 * 多変量データの解析に関する関数を提供します。
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "statcpp/linear_regression.hpp"  // detail::validate_matrix_structure 用

namespace statcpp {

// ============================================================================
// Covariance Matrix
// ============================================================================

/**
 * @brief 標本共分散行列を計算
 *
 * 多変量データの共分散行列を計算します。
 * データは行=観測、列=変数の形式で渡します。
 *
 * @param data 行列データ（行=観測、列=変数）
 * @return 共分散行列（p x p）
 * @throws std::invalid_argument データが空の場合、行の列数が一致しない場合、または観測数が2未満の場合
 */
inline std::vector<std::vector<double>> covariance_matrix(
    const std::vector<std::vector<double>>& data)
{
    // 行列構造の検証（空チェックと列数の一貫性）
    detail::validate_matrix_structure(data, "covariance_matrix");

    std::size_t n = data.size();      // 観測数
    std::size_t p = data[0].size();   // 変数数

    if (n < 2) {
        throw std::invalid_argument("statcpp::covariance_matrix: need at least 2 observations");
    }

    // 各変数の平均を計算
    std::vector<double> means(p, 0.0);
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            means[j] += data[i][j];
        }
        means[j] /= static_cast<double>(n);
    }

    // 共分散行列を計算
    std::vector<std::vector<double>> cov(p, std::vector<double>(p, 0.0));
    for (std::size_t j1 = 0; j1 < p; ++j1) {
        for (std::size_t j2 = j1; j2 < p; ++j2) {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                sum += (data[i][j1] - means[j1]) * (data[i][j2] - means[j2]);
            }
            cov[j1][j2] = sum / static_cast<double>(n - 1);
            cov[j2][j1] = cov[j1][j2];  // 対称行列
        }
    }

    return cov;
}

// ============================================================================
// Correlation Matrix
// ============================================================================

/**
 * @brief ピアソン相関行列を計算
 *
 * 多変量データの相関行列を計算します。
 * 共分散行列を標準化して相関係数を求めます。
 *
 * @param data 行列データ（行=観測、列=変数）
 * @return 相関行列（p x p）
 * @throws std::invalid_argument データが空の場合、行の列数が一致しない場合、観測数が2未満の場合、または分散が0の変数がある場合
 */
inline std::vector<std::vector<double>> correlation_matrix(
    const std::vector<std::vector<double>>& data)
{
    // 行列構造の検証
    detail::validate_matrix_structure(data, "correlation_matrix");

    std::size_t n = data.size();
    std::size_t p = data[0].size();

    if (n < 2) {
        throw std::invalid_argument("statcpp::correlation_matrix: need at least 2 observations");
    }

    // 共分散行列を計算
    auto cov = covariance_matrix(data);

    // 標準偏差を計算
    std::vector<double> stddevs(p);
    for (std::size_t j = 0; j < p; ++j) {
        stddevs[j] = std::sqrt(cov[j][j]);
        if (stddevs[j] == 0.0) {
            throw std::invalid_argument("statcpp::correlation_matrix: zero variance variable");
        }
    }

    // 相関行列に変換
    std::vector<std::vector<double>> corr(p, std::vector<double>(p, 0.0));
    for (std::size_t j1 = 0; j1 < p; ++j1) {
        for (std::size_t j2 = 0; j2 < p; ++j2) {
            corr[j1][j2] = cov[j1][j2] / (stddevs[j1] * stddevs[j2]);
        }
    }

    return corr;
}

// ============================================================================
// Standardization (Z-score)
// ============================================================================

/**
 * @brief Z スコア標準化
 *
 * 各変数を平均0、標準偏差1に標準化します。
 * データは行=観測、列=変数の形式で渡します。
 *
 * @param data 行列データ（行=観測、列=変数）
 * @return 標準化されたデータ
 * @throws std::invalid_argument データが空の場合、行の列数が一致しない場合、観測数が2未満の場合、または分散が0の変数がある場合
 */
inline std::vector<std::vector<double>> standardize(
    const std::vector<std::vector<double>>& data)
{
    // 行列構造の検証
    detail::validate_matrix_structure(data, "standardize");

    std::size_t n = data.size();
    std::size_t p = data[0].size();

    if (n < 2) {
        throw std::invalid_argument("statcpp::standardize: need at least 2 observations");
    }

    // 各変数の平均と標準偏差を計算
    std::vector<double> means(p, 0.0);
    std::vector<double> stddevs(p, 0.0);

    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            means[j] += data[i][j];
        }
        means[j] /= static_cast<double>(n);

        for (std::size_t i = 0; i < n; ++i) {
            double diff = data[i][j] - means[j];
            stddevs[j] += diff * diff;
        }
        stddevs[j] = std::sqrt(stddevs[j] / static_cast<double>(n - 1));

        if (stddevs[j] == 0.0) {
            throw std::invalid_argument("statcpp::standardize: zero variance variable");
        }
    }

    // 標準化
    std::vector<std::vector<double>> result(n, std::vector<double>(p));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            result[i][j] = (data[i][j] - means[j]) / stddevs[j];
        }
    }

    return result;
}

// ============================================================================
// Min-Max Scaling
// ============================================================================

/**
 * @brief Min-Max 正規化（0-1 スケーリング）
 *
 * 各変数を [0, 1] の範囲にスケーリングします。
 * データは行=観測、列=変数の形式で渡します。
 *
 * @param data 行列データ（行=観測、列=変数）
 * @return スケーリングされたデータ
 * @throws std::invalid_argument データが空の場合、行の列数が一致しない場合、または範囲が0の変数がある場合
 */
inline std::vector<std::vector<double>> min_max_scale(
    const std::vector<std::vector<double>>& data)
{
    // 行列構造の検証
    detail::validate_matrix_structure(data, "min_max_scale");

    std::size_t n = data.size();
    std::size_t p = data[0].size();

    // 各変数の最小値と最大値を計算
    std::vector<double> mins(p, std::numeric_limits<double>::max());
    std::vector<double> maxs(p, std::numeric_limits<double>::lowest());

    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            mins[j] = std::min(mins[j], data[i][j]);
            maxs[j] = std::max(maxs[j], data[i][j]);
        }

        if (maxs[j] == mins[j]) {
            throw std::invalid_argument("statcpp::min_max_scale: zero range variable");
        }
    }

    // スケーリング
    std::vector<std::vector<double>> result(n, std::vector<double>(p));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            result[i][j] = (data[i][j] - mins[j]) / (maxs[j] - mins[j]);
        }
    }

    return result;
}

// ============================================================================
// Principal Component Analysis (PCA)
// ============================================================================

/**
 * @brief PCA の結果
 *
 * 主成分分析の結果を保持します。
 */
struct pca_result {
    std::vector<std::vector<double>> components;  ///< 主成分（固有ベクトル）p x n_components
    std::vector<double> explained_variance;       ///< 説明分散（固有値）
    std::vector<double> explained_variance_ratio; ///< 寄与率
};

/**
 * @brief べき乗法で最大固有値と固有ベクトルを求める
 *
 * 反復計算により行列の最大固有値と対応する固有ベクトルを求めます。
 *
 * @param matrix 対称行列
 * @param max_iter 最大反復回数（デフォルト: 1000）
 * @param tol 収束判定の閾値（デフォルト: 1e-10）
 * @return (固有値, 固有ベクトル)のペア
 */
inline std::pair<double, std::vector<double>> power_iteration(
    const std::vector<std::vector<double>>& matrix,
    std::size_t max_iter = 1000,
    double tol = 1e-10)
{
    std::size_t n = matrix.size();
    std::vector<double> v(n, 1.0 / std::sqrt(static_cast<double>(n)));

    double eigenvalue = 0.0;

    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        // Av を計算
        std::vector<double> Av(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                Av[i] += matrix[i][j] * v[j];
            }
        }

        // ノルムを計算
        double norm = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            norm += Av[i] * Av[i];
        }
        norm = std::sqrt(norm);

        if (norm == 0.0) break;

        // 固有値の推定
        double new_eigenvalue = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            new_eigenvalue += v[i] * Av[i];
        }

        // 正規化
        for (std::size_t i = 0; i < n; ++i) {
            v[i] = Av[i] / norm;
        }

        // 収束判定
        if (std::abs(new_eigenvalue - eigenvalue) < tol) {
            eigenvalue = new_eigenvalue;
            break;
        }
        eigenvalue = new_eigenvalue;
    }

    return {eigenvalue, v};
}

/**
 * @brief 主成分分析
 *
 * 共分散行列の固有値分解により主成分を求めます。
 * デフレーション法により複数の主成分を計算します。
 *
 * @param data 行列データ（行=観測、列=変数）
 * @param n_components 抽出する主成分の数
 * @return PCA の結果
 * @throws std::invalid_argument データが空の場合、または行の列数が一致しない場合
 */
inline pca_result pca(const std::vector<std::vector<double>>& data,
                      std::size_t n_components)
{
    // 行列構造の検証
    detail::validate_matrix_structure(data, "pca");

    std::size_t p = data[0].size();

    if (n_components > p) {
        n_components = p;
    }

    // 共分散行列を計算
    auto cov = covariance_matrix(data);

    pca_result result;
    result.components.resize(p, std::vector<double>(n_components));
    result.explained_variance.resize(n_components);
    result.explained_variance_ratio.resize(n_components);

    // 総分散（トレース）
    double total_variance = 0.0;
    for (std::size_t j = 0; j < p; ++j) {
        total_variance += cov[j][j];
    }

    // 各主成分を計算（デフレーション法）
    auto working_cov = cov;

    for (std::size_t k = 0; k < n_components; ++k) {
        auto [eigenvalue, eigenvector] = power_iteration(working_cov);

        result.explained_variance[k] = eigenvalue;
        result.explained_variance_ratio[k] = eigenvalue / total_variance;

        for (std::size_t j = 0; j < p; ++j) {
            result.components[j][k] = eigenvector[j];
        }

        // デフレーション：処理済みの固有値成分を除去
        for (std::size_t i = 0; i < p; ++i) {
            for (std::size_t j = 0; j < p; ++j) {
                working_cov[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
            }
        }
    }

    return result;
}

/**
 * @brief データを主成分空間に射影
 *
 * PCAの結果を用いてデータを主成分空間に変換します。
 *
 * @param data 行列データ（行=観測、列=変数）
 * @param pca PCA の結果
 * @return 主成分空間に射影されたデータ（n x n_components）
 * @throws std::invalid_argument データが空の場合、PCAの成分が空の場合、またはデータとPCAの次元が一致しない場合
 */
inline std::vector<std::vector<double>> pca_transform(
    const std::vector<std::vector<double>>& data,
    const pca_result& pca)
{
    // 行列構造の検証
    detail::validate_matrix_structure(data, "pca_transform");

    // PCA components の構造も検証
    if (pca.components.empty() || pca.components[0].empty()) {
        throw std::invalid_argument("statcpp::pca_transform: pca.components is empty");
    }

    std::size_t n = data.size();
    std::size_t p = data[0].size();
    std::size_t n_components = pca.components[0].size();

    // データとPCA componentsの次元が一致するかチェック
    if (p != pca.components.size()) {
        throw std::invalid_argument("statcpp::pca_transform: dimension mismatch between data and pca.components");
    }

    // 平均を計算
    std::vector<double> means(p, 0.0);
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            means[j] += data[i][j];
        }
        means[j] /= static_cast<double>(n);
    }

    // 射影
    std::vector<std::vector<double>> result(n, std::vector<double>(n_components, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n_components; ++k) {
            for (std::size_t j = 0; j < p; ++j) {
                result[i][k] += (data[i][j] - means[j]) * pca.components[j][k];
            }
        }
    }

    return result;
}

} // namespace statcpp
