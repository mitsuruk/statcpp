/**
 * @file linear_regression.hpp
 * @brief 線形回帰分析
 *
 * 単回帰、重回帰、多項式回帰などの線形回帰分析機能を提供します。
 * 予測、信頼区間、残差診断、多重共線性診断などの機能を含みます。
 */

#pragma once

#include "statcpp/basic_statistics.hpp"
#include "statcpp/continuous_distributions.hpp"
#include "statcpp/correlation_covariance.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace statcpp {

// ============================================================================
// Linear Regression Result Structures
// ============================================================================

/**
 * @brief 単回帰分析の結果を格納する構造体
 *
 * 単回帰分析 y = b0 + b1*x の結果として、回帰係数、標準誤差、
 * 検定統計量、決定係数などを保持します。
 */
struct simple_regression_result {
    double intercept;           ///< 切片 (b0)
    double slope;               ///< 傾き (b1)
    double intercept_se;        ///< 切片の標準誤差
    double slope_se;            ///< 傾きの標準誤差
    double intercept_t;         ///< 切片のt統計量
    double slope_t;             ///< 傾きのt統計量
    double intercept_p;         ///< 切片のp値
    double slope_p;             ///< 傾きのp値
    double r_squared;           ///< 決定係数 R^2
    double adj_r_squared;       ///< 自由度調整済みR^2
    double residual_se;         ///< 残差の標準誤差
    double f_statistic;         ///< F統計量
    double f_p_value;           ///< F検定のp値
    double df_regression;       ///< 回帰の自由度
    double df_residual;         ///< 残差の自由度
    double ss_total;            ///< 全平方和
    double ss_regression;       ///< 回帰平方和
    double ss_residual;         ///< 残差平方和
};

/**
 * @brief 重回帰分析の結果を格納する構造体
 *
 * 重回帰分析 y = b0 + b1*x1 + ... + bp*xp の結果として、
 * 各回帰係数、標準誤差、検定統計量、決定係数などを保持します。
 */
struct multiple_regression_result {
    std::vector<double> coefficients;       ///< 回帰係数 (b0, b1, ..., bp)
    std::vector<double> coefficient_se;     ///< 係数の標準誤差
    std::vector<double> t_statistics;       ///< t統計量
    std::vector<double> p_values;           ///< p値
    double r_squared;                       ///< 決定係数 R^2
    double adj_r_squared;                   ///< 自由度調整済みR^2
    double residual_se;                     ///< 残差の標準誤差
    double f_statistic;                     ///< F統計量
    double f_p_value;                       ///< F検定のp値
    double df_regression;                   ///< 回帰の自由度
    double df_residual;                     ///< 残差の自由度
    double ss_total;                        ///< 全平方和
    double ss_regression;                   ///< 回帰平方和
    double ss_residual;                     ///< 残差平方和
};

/**
 * @brief 予測区間の結果を格納する構造体
 *
 * 回帰モデルによる予測値とその信頼区間（または予測区間）を保持します。
 */
struct prediction_interval {
    double prediction;      ///< 予測値
    double lower;           ///< 下限
    double upper;           ///< 上限
    double se_prediction;   ///< 予測値の標準誤差
};

/**
 * @brief 残差診断の結果を格納する構造体
 *
 * 回帰モデルの残差に関する各種診断統計量を保持します。
 * 異常値や影響点の検出、残差の自己相関の検定などに使用します。
 */
struct residual_diagnostics {
    std::vector<double> residuals;              ///< 残差
    std::vector<double> standardized_residuals; ///< 標準化残差
    std::vector<double> studentized_residuals;  ///< スチューデント化残差
    std::vector<double> hat_values;             ///< てこ比 (leverage)
    std::vector<double> cooks_distance;         ///< クックの距離
    double durbin_watson;                       ///< ダービン・ワトソン統計量
};

// ============================================================================
// Simple Linear Regression (単回帰分析)
// ============================================================================

/**
 * @brief 単回帰分析を実行する
 *
 * 最小二乗法により単回帰モデル y = b0 + b1*x を推定します。
 * 回帰係数、標準誤差、t検定、F検定、決定係数などを計算します。
 *
 * @tparam IteratorX 説明変数のイテレータ型
 * @tparam IteratorY 目的変数のイテレータ型
 * @param x_first 説明変数の開始イテレータ
 * @param x_last 説明変数の終了イテレータ
 * @param y_first 目的変数の開始イテレータ
 * @param y_last 目的変数の終了イテレータ
 * @return simple_regression_result 回帰分析の結果
 * @throws std::invalid_argument xとyの長さが異なる場合
 * @throws std::invalid_argument 観測数が3未満の場合
 * @throws std::invalid_argument xの分散が0の場合
 */
template <typename IteratorX, typename IteratorY>
simple_regression_result simple_linear_regression(IteratorX x_first, IteratorX x_last,
                                                   IteratorY y_first, IteratorY y_last)
{
    auto n_x = statcpp::count(x_first, x_last);
    auto n_y = statcpp::count(y_first, y_last);

    if (n_x != n_y) {
        throw std::invalid_argument("statcpp::simple_linear_regression: x and y must have same length");
    }
    if (n_x < 3) {
        throw std::invalid_argument("statcpp::simple_linear_regression: need at least 3 observations");
    }

    std::size_t n = n_x;
    double n_d = static_cast<double>(n);

    // 平均値を計算
    double mean_x = statcpp::mean(x_first, x_last);
    double mean_y = statcpp::mean(y_first, y_last);

    // Sxx, Syy, Sxy を計算
    double sxx = 0.0, syy = 0.0, sxy = 0.0;
    auto it_x = x_first;
    auto it_y = y_first;
    for (; it_x != x_last; ++it_x, ++it_y) {
        double dx = static_cast<double>(*it_x) - mean_x;
        double dy = static_cast<double>(*it_y) - mean_y;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }

    if (sxx == 0.0) {
        throw std::invalid_argument("statcpp::simple_linear_regression: zero variance in x");
    }

    // 回帰係数
    double slope = sxy / sxx;
    double intercept = mean_y - slope * mean_x;

    // 平方和
    double ss_total = syy;
    double ss_regression = slope * sxy;
    double ss_residual = ss_total - ss_regression;

    if (ss_total == 0.0) {
        throw std::invalid_argument("statcpp::simple_linear_regression: zero variance in y (constant response)");
    }

    // 残差分散と標準誤差
    double df_reg = 1.0;
    double df_res = n_d - 2.0;
    double mse = ss_residual / df_res;
    double residual_se = std::sqrt(mse);

    // 係数の標準誤差
    double slope_se = residual_se / std::sqrt(sxx);
    double intercept_se = residual_se * std::sqrt(1.0 / n_d + mean_x * mean_x / sxx);

    // t統計量とp値
    double slope_t = slope / slope_se;
    double intercept_t = intercept / intercept_se;
    double slope_p = 2.0 * (1.0 - t_cdf(std::abs(slope_t), df_res));
    double intercept_p = 2.0 * (1.0 - t_cdf(std::abs(intercept_t), df_res));

    // 決定係数
    double r_squared = ss_regression / ss_total;
    double adj_r_squared = 1.0 - (1.0 - r_squared) * (n_d - 1.0) / df_res;

    // F統計量
    double f_statistic = (ss_regression / df_reg) / mse;
    double f_p_value = 1.0 - f_cdf(f_statistic, df_reg, df_res);

    return {
        intercept, slope,
        intercept_se, slope_se,
        intercept_t, slope_t,
        intercept_p, slope_p,
        r_squared, adj_r_squared,
        residual_se,
        f_statistic, f_p_value,
        df_reg, df_res,
        ss_total, ss_regression, ss_residual
    };
}

// ============================================================================
// Multiple Linear Regression (重回帰分析)
// ============================================================================

namespace detail {

/**
 * @brief 2D行列の構造を検証する
 *
 * 行列が空でないこと、全行が同じ列数を持つことを確認します。
 *
 * @param data 検証対象の行列
 * @param func_name エラーメッセージに含める関数名
 * @throws std::invalid_argument 行列が空の場合
 * @throws std::invalid_argument 最初の行が空の場合
 * @throws std::invalid_argument 行間で列数が異なる場合
 */
inline void validate_matrix_structure(const std::vector<std::vector<double>>& data,
                                      const char* func_name)
{
    if (data.empty()) {
        std::string msg = "statcpp::";
        msg += func_name;
        msg += ": empty data";
        throw std::invalid_argument(msg);
    }

    std::size_t p = data[0].size();
    if (p == 0) {
        std::string msg = "statcpp::";
        msg += func_name;
        msg += ": first row is empty (0 columns)";
        throw std::invalid_argument(msg);
    }

    // 全ての行が同じ列数を持つことを確認
    for (std::size_t i = 1; i < data.size(); ++i) {
        if (data[i].size() != p) {
            std::string msg = "statcpp::";
            msg += func_name;
            msg += ": inconsistent row dimensions (row 0 has ";
            msg += std::to_string(p);
            msg += " columns, but row ";
            msg += std::to_string(i);
            msg += " has ";
            msg += std::to_string(data[i].size());
            msg += " columns)";
            throw std::invalid_argument(msg);
        }
    }
}

/**
 * @brief Xデータに切片列が含まれていないかをチェックする
 *
 * 最初の列が全て1.0の場合、ユーザーが誤って切片列を含めている可能性があるため警告します。
 * 切片は関数内で自動追加されるため、ユーザーが含める必要はありません。
 *
 * @param X 説明変数の行列
 * @param func_name エラーメッセージに含める関数名
 * @throws std::invalid_argument 最初の列が全て1.0の場合
 */
inline void validate_no_intercept_column(const std::vector<std::vector<double>>& X, const char* func_name)
{
    if (X.empty()) {
        return;
    }

    std::size_t n = X.size();
    std::size_t p = X[0].size();

    // 最初の列がすべて1.0かチェック
    if (p > 0) {
        bool all_ones = true;
        for (std::size_t i = 0; i < n; ++i) {
            if (X[i].size() == 0 || std::abs(X[i][0] - 1.0) > 1e-10) {
                all_ones = false;
                break;
            }
        }

        if (all_ones) {
            std::string msg = "statcpp::";
            msg += func_name;
            msg += ": X should not contain intercept column (all 1s in first column detected). ";
            msg += "The intercept is added automatically.";
            throw std::invalid_argument(msg);
        }
    }
}

// 簡易的な行列演算（小規模行列向け）

/**
 * @brief 転置行列を計算する
 *
 * @param A 入力行列
 * @return std::vector<std::vector<double>> 転置行列
 * @throws std::invalid_argument 行間で列数が異なる場合
 */
inline std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& A)
{
    if (A.empty()) return {};

    // 行列構造の検証（全行が同じサイズか確認）
    std::size_t rows = A.size();
    std::size_t cols = A[0].size();
    for (std::size_t i = 1; i < rows; ++i) {
        if (A[i].size() != cols) {
            throw std::invalid_argument("statcpp::detail::transpose: inconsistent row dimensions");
        }
    }

    std::vector<std::vector<double>> result(cols, std::vector<double>(rows));
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            result[j][i] = A[i][j];
        }
    }
    return result;
}

/**
 * @brief 行列の積を計算する
 *
 * @param A 左側の行列 (m x n)
 * @param B 右側の行列 (n x p)
 * @return std::vector<std::vector<double>> 積行列 (m x p)
 * @throws std::invalid_argument 行列の次元が適合しない場合
 * @throws std::invalid_argument 行列の行サイズが一貫していない場合
 */
inline std::vector<std::vector<double>> matrix_multiply(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B)
{
    if (A.empty() || B.empty()) return {};
    std::size_t m = A.size();
    std::size_t n = A[0].size();
    std::size_t p = B[0].size();

    if (n != B.size()) {
        throw std::invalid_argument("statcpp::detail::matrix_multiply: incompatible dimensions");
    }

    // 各行のサイズが一貫しているか確認
    for (std::size_t i = 0; i < m; ++i) {
        if (A[i].size() != n) {
            throw std::invalid_argument("statcpp::detail::matrix_multiply: matrix A has inconsistent row dimensions");
        }
    }
    for (std::size_t k = 0; k < n; ++k) {
        if (B[k].size() != p) {
            throw std::invalid_argument("statcpp::detail::matrix_multiply: matrix B has inconsistent row dimensions");
        }
    }

    std::vector<std::vector<double>> result(m, std::vector<double>(p, 0.0));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            for (std::size_t k = 0; k < n; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

/**
 * @brief 行列とベクトルの積を計算する
 *
 * @param A 行列 (m x n)
 * @param v ベクトル (n次元)
 * @return std::vector<double> 結果ベクトル (m次元)
 * @throws std::invalid_argument 行列とベクトルの次元が適合しない場合
 */
inline std::vector<double> matrix_vector_multiply(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& v)
{
    if (A.empty()) return {};
    std::size_t m = A.size();
    std::size_t n = A[0].size();

    if (n != v.size()) {
        throw std::invalid_argument("statcpp::detail::matrix_vector_multiply: incompatible dimensions");
    }

    std::vector<double> result(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            result[i] += A[i][j] * v[j];
        }
    }
    return result;
}

/**
 * @brief Cholesky分解を実行する
 *
 * 正定値対称行列 A に対して L * L^T = A となる下三角行列 L を計算します。
 *
 * @param A 正定値対称行列
 * @return std::vector<std::vector<double>> 下三角行列 L
 * @throws std::runtime_error 行列が正定値でない場合
 */
inline std::vector<std::vector<double>> cholesky(const std::vector<std::vector<double>>& A)
{
    std::size_t n = A.size();
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < j; ++k) {
                sum += L[i][k] * L[j][k];
            }
            if (i == j) {
                double val = A[i][i] - sum;
                if (val <= 0.0) {
                    throw std::runtime_error("statcpp::detail::cholesky: matrix is not positive definite");
                }
                L[i][j] = std::sqrt(val);
            } else {
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }
    return L;
}

/**
 * @brief Cholesky分解を用いて連立方程式を解く
 *
 * A * x = b を解きます（A = L * L^T）。
 * 前進代入と後退代入により効率的に解を求めます。
 *
 * @param L Cholesky分解で得られた下三角行列
 * @param b 右辺ベクトル
 * @return std::vector<double> 解ベクトル x
 */
inline std::vector<double> solve_cholesky(
    const std::vector<std::vector<double>>& L,
    const std::vector<double>& b)
{
    std::size_t n = L.size();

    // 前進代入: L * y = b
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < i; ++j) {
            sum += L[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i][i];
    }

    // 後退代入: L^T * x = y
    std::vector<double> x(n);
    for (std::size_t i = n; i > 0; --i) {
        std::size_t idx = i - 1;
        double sum = 0.0;
        for (std::size_t j = idx + 1; j < n; ++j) {
            sum += L[j][idx] * x[j];
        }
        x[idx] = (y[idx] - sum) / L[idx][idx];
    }
    return x;
}

/**
 * @brief Cholesky分解を用いて逆行列を計算する
 *
 * @param L Cholesky分解で得られた下三角行列
 * @return std::vector<std::vector<double>> 元の行列の逆行列
 */
inline std::vector<std::vector<double>> inverse_cholesky(
    const std::vector<std::vector<double>>& L)
{
    std::size_t n = L.size();
    std::vector<std::vector<double>> inv(n, std::vector<double>(n, 0.0));

    // 各列を解く
    for (std::size_t j = 0; j < n; ++j) {
        std::vector<double> e(n, 0.0);
        e[j] = 1.0;
        std::vector<double> col = solve_cholesky(L, e);
        for (std::size_t i = 0; i < n; ++i) {
            inv[i][j] = col[i];
        }
    }
    return inv;
}

} // namespace detail

/**
 * @brief 重回帰分析を実行する
 *
 * 最小二乗法により重回帰モデル y = b0 + b1*x1 + ... + bp*xp を推定します。
 * 切片は自動的に追加されるため、Xには切片列を含めないでください。
 *
 * @param X 説明変数の行列 (n x p)。各行が1観測、各列が1説明変数
 * @param y 目的変数のベクトル (n次元)
 * @return multiple_regression_result 回帰分析の結果
 * @throws std::invalid_argument データが空の場合
 * @throws std::invalid_argument XとYの観測数が異なる場合
 * @throws std::invalid_argument 説明変数の数が観測数以上の場合
 * @throws std::invalid_argument Xに切片列（全て1の列）が含まれている場合
 */
inline multiple_regression_result multiple_linear_regression(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y)
{
    // X に切片列が含まれていないことを確認
    detail::validate_no_intercept_column(X, "multiple_linear_regression");

    std::size_t n = X.size();
    if (n == 0) {
        throw std::invalid_argument("statcpp::multiple_linear_regression: empty data");
    }
    if (n != y.size()) {
        throw std::invalid_argument("statcpp::multiple_linear_regression: X and y must have same number of observations");
    }

    std::size_t p = X[0].size();  // 説明変数の数（切片を除く）
    for (const auto& row : X) {
        if (row.size() != p) {
            throw std::invalid_argument("statcpp::multiple_linear_regression: inconsistent number of predictors");
        }
    }

    std::size_t p_full = p + 1;  // 切片を含む係数の数
    if (n <= p_full) {
        throw std::invalid_argument("statcpp::multiple_linear_regression: need more observations than predictors");
    }

    double n_d = static_cast<double>(n);

    // 切片項を追加した設計行列を作成
    std::vector<std::vector<double>> X_design(n, std::vector<double>(p_full));
    for (std::size_t i = 0; i < n; ++i) {
        X_design[i][0] = 1.0;  // 切片
        for (std::size_t j = 0; j < p; ++j) {
            X_design[i][j + 1] = X[i][j];
        }
    }

    // X^T * X を計算
    auto Xt = detail::transpose(X_design);
    auto XtX = detail::matrix_multiply(Xt, X_design);

    // X^T * y を計算
    auto Xty = detail::matrix_vector_multiply(Xt, y);

    // Cholesky分解で係数を求める
    auto L = detail::cholesky(XtX);
    auto coefficients = detail::solve_cholesky(L, Xty);

    // (X^T X)^{-1} を計算
    auto XtX_inv = detail::inverse_cholesky(L);

    // 予測値と残差を計算
    std::vector<double> y_hat(n);
    std::vector<double> residuals(n);
    double mean_y = statcpp::mean(y.begin(), y.end());

    double ss_total = 0.0;
    double ss_residual = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        double pred = 0.0;
        for (std::size_t j = 0; j < p_full; ++j) {
            pred += X_design[i][j] * coefficients[j];
        }
        y_hat[i] = pred;
        residuals[i] = y[i] - pred;

        ss_total += (y[i] - mean_y) * (y[i] - mean_y);
        ss_residual += residuals[i] * residuals[i];
    }

    double ss_regression = ss_total - ss_residual;

    if (ss_total == 0.0) {
        throw std::invalid_argument("statcpp::multiple_linear_regression: zero variance in y (constant response)");
    }

    // 自由度
    double df_reg = static_cast<double>(p);
    double df_res = n_d - static_cast<double>(p_full);

    // 残差分散
    double mse = ss_residual / df_res;
    double residual_se = std::sqrt(mse);

    // 係数の標準誤差、t統計量、p値
    std::vector<double> coefficient_se(p_full);
    std::vector<double> t_statistics(p_full);
    std::vector<double> p_values(p_full);

    for (std::size_t j = 0; j < p_full; ++j) {
        coefficient_se[j] = std::sqrt(mse * XtX_inv[j][j]);
        t_statistics[j] = coefficients[j] / coefficient_se[j];
        p_values[j] = 2.0 * (1.0 - t_cdf(std::abs(t_statistics[j]), df_res));
    }

    // 決定係数
    double r_squared = ss_regression / ss_total;
    double adj_r_squared = 1.0 - (1.0 - r_squared) * (n_d - 1.0) / df_res;

    // F統計量
    double f_statistic = (ss_regression / df_reg) / mse;
    double f_p_value = 1.0 - f_cdf(f_statistic, df_reg, df_res);

    return {
        coefficients, coefficient_se, t_statistics, p_values,
        r_squared, adj_r_squared,
        residual_se,
        f_statistic, f_p_value,
        df_reg, df_res,
        ss_total, ss_regression, ss_residual
    };
}

// ============================================================================
// Prediction (予測)
// ============================================================================

/**
 * @brief 単回帰モデルによる予測を行う
 *
 * @param model 単回帰分析の結果
 * @param x 説明変数の値
 * @return double 予測値
 */
inline double predict(const simple_regression_result& model, double x)
{
    return model.intercept + model.slope * x;
}

/**
 * @brief 重回帰モデルによる予測を行う
 *
 * @param model 重回帰分析の結果
 * @param x 説明変数のベクトル（切片は含めない）
 * @return double 予測値
 * @throws std::invalid_argument xの次元がモデルの説明変数の数と一致しない場合
 */
inline double predict(const multiple_regression_result& model, const std::vector<double>& x)
{
    if (x.size() + 1 != model.coefficients.size()) {
        throw std::invalid_argument("statcpp::predict: x dimension mismatch");
    }

    double pred = model.coefficients[0];  // 切片
    for (std::size_t i = 0; i < x.size(); ++i) {
        pred += model.coefficients[i + 1] * x[i];
    }
    return pred;
}

// ============================================================================
// Prediction Interval (予測区間)
// ============================================================================

/**
 * @brief 単回帰モデルの予測区間を計算する
 *
 * 新しい観測値に対する予測区間を計算します。
 * 予測区間は、将来の個別の観測値がこの区間内に入る確率を示します。
 *
 * @tparam IteratorX 説明変数のイテレータ型
 * @param model 単回帰分析の結果
 * @param x_first 元データの説明変数の開始イテレータ
 * @param x_last 元データの説明変数の終了イテレータ
 * @param x_new 予測したい点のx値
 * @param confidence 信頼水準（デフォルト: 0.95）
 * @return prediction_interval 予測値と予測区間
 * @throws std::invalid_argument confidenceが(0, 1)の範囲外の場合
 */
template <typename IteratorX>
prediction_interval prediction_interval_simple(
    const simple_regression_result& model,
    IteratorX x_first, IteratorX x_last,
    double x_new,
    double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::prediction_interval_simple: confidence must be in (0, 1)");
    }

    auto n = statcpp::count(x_first, x_last);
    double n_d = static_cast<double>(n);
    double mean_x = statcpp::mean(x_first, x_last);

    // Sxx を計算
    double sxx = 0.0;
    for (auto it = x_first; it != x_last; ++it) {
        double dx = static_cast<double>(*it) - mean_x;
        sxx += dx * dx;
    }

    double y_hat = predict(model, x_new);

    // 予測値の標準誤差（新しい観測に対する）
    double dx_new = x_new - mean_x;
    double se_pred = model.residual_se * std::sqrt(1.0 + 1.0 / n_d + dx_new * dx_new / sxx);

    double t_crit = t_quantile(1.0 - (1.0 - confidence) / 2.0, model.df_residual);
    double margin = t_crit * se_pred;

    return {y_hat, y_hat - margin, y_hat + margin, se_pred};
}

/**
 * @brief 単回帰モデルの平均値の信頼区間を計算する
 *
 * 特定のx値における平均応答の信頼区間を計算します。
 * 信頼区間は、真の回帰直線がこの区間内にある確率を示します。
 *
 * @tparam IteratorX 説明変数のイテレータ型
 * @param model 単回帰分析の結果
 * @param x_first 元データの説明変数の開始イテレータ
 * @param x_last 元データの説明変数の終了イテレータ
 * @param x_new 予測したい点のx値
 * @param confidence 信頼水準（デフォルト: 0.95）
 * @return prediction_interval 予測値と信頼区間
 * @throws std::invalid_argument confidenceが(0, 1)の範囲外の場合
 */
template <typename IteratorX>
prediction_interval confidence_interval_mean(
    const simple_regression_result& model,
    IteratorX x_first, IteratorX x_last,
    double x_new,
    double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::confidence_interval_mean: confidence must be in (0, 1)");
    }

    auto n = statcpp::count(x_first, x_last);
    double n_d = static_cast<double>(n);
    double mean_x = statcpp::mean(x_first, x_last);

    // Sxx を計算
    double sxx = 0.0;
    for (auto it = x_first; it != x_last; ++it) {
        double dx = static_cast<double>(*it) - mean_x;
        sxx += dx * dx;
    }

    double y_hat = predict(model, x_new);

    // 平均予測値の標準誤差
    double dx_new = x_new - mean_x;
    double se_mean = model.residual_se * std::sqrt(1.0 / n_d + dx_new * dx_new / sxx);

    double t_crit = t_quantile(1.0 - (1.0 - confidence) / 2.0, model.df_residual);
    double margin = t_crit * se_mean;

    return {y_hat, y_hat - margin, y_hat + margin, se_mean};
}

// ============================================================================
// Residual Diagnostics (残差診断)
// ============================================================================

/**
 * @brief 単回帰モデルの残差診断を行う
 *
 * 残差、標準化残差、スチューデント化残差、てこ比、クックの距離、
 * ダービン・ワトソン統計量を計算します。
 *
 * @tparam IteratorX 説明変数のイテレータ型
 * @tparam IteratorY 目的変数のイテレータ型
 * @param model 単回帰分析の結果
 * @param x_first 説明変数の開始イテレータ
 * @param x_last 説明変数の終了イテレータ
 * @param y_first 目的変数の開始イテレータ
 * @param y_last 目的変数の終了イテレータ
 * @return residual_diagnostics 残差診断の結果
 * @throws std::invalid_argument xとyの長さが異なる場合
 */
template <typename IteratorX, typename IteratorY>
residual_diagnostics compute_residual_diagnostics(
    const simple_regression_result& model,
    IteratorX x_first, IteratorX x_last,
    IteratorY y_first, IteratorY y_last)
{
    auto n = statcpp::count(x_first, x_last);
    if (n != statcpp::count(y_first, y_last)) {
        throw std::invalid_argument("statcpp::compute_residual_diagnostics: x and y must have same length");
    }

    double n_d = static_cast<double>(n);
    double mean_x = statcpp::mean(x_first, x_last);

    // Sxx を計算
    double sxx = 0.0;
    for (auto it = x_first; it != x_last; ++it) {
        double dx = static_cast<double>(*it) - mean_x;
        sxx += dx * dx;
    }

    std::vector<double> residuals(n);
    std::vector<double> hat_values(n);
    std::vector<double> standardized_residuals(n);

    auto it_x = x_first;
    auto it_y = y_first;
    for (std::size_t i = 0; it_x != x_last; ++it_x, ++it_y, ++i) {
        double x_i = static_cast<double>(*it_x);
        double y_i = static_cast<double>(*it_y);
        double y_hat = predict(model, x_i);
        residuals[i] = y_i - y_hat;

        // てこ比 h_ii = 1/n + (x_i - mean_x)^2 / Sxx
        double dx = x_i - mean_x;
        hat_values[i] = 1.0 / n_d + dx * dx / sxx;

        // 標準化残差
        standardized_residuals[i] = residuals[i] / model.residual_se;
    }

    // スチューデント化残差とクックの距離
    std::vector<double> studentized_residuals(n);
    std::vector<double> cooks_distance(n);
    double p = 2.0;  // 係数の数（切片 + 傾き）

    for (std::size_t i = 0; i < n; ++i) {
        double h_i = hat_values[i];
        double se_i = model.residual_se * std::sqrt(1.0 - h_i);
        studentized_residuals[i] = (se_i > 0.0) ? residuals[i] / se_i : 0.0;
        cooks_distance[i] = (standardized_residuals[i] * standardized_residuals[i] / p)
                          * (h_i / (1.0 - h_i));
    }

    // ダービン・ワトソン統計量
    double dw_num = 0.0;
    double dw_den = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        dw_den += residuals[i] * residuals[i];
        if (i > 0) {
            double diff = residuals[i] - residuals[i - 1];
            dw_num += diff * diff;
        }
    }
    double durbin_watson = (dw_den > 0.0) ? dw_num / dw_den : 0.0;

    return {residuals, standardized_residuals, studentized_residuals,
            hat_values, cooks_distance, durbin_watson};
}

/**
 * @brief 重回帰モデルの残差診断を行う
 *
 * 残差、標準化残差、スチューデント化残差、てこ比、クックの距離、
 * ダービン・ワトソン統計量を計算します。
 *
 * @param model 重回帰分析の結果
 * @param X 説明変数の行列
 * @param y 目的変数のベクトル
 * @return residual_diagnostics 残差診断の結果
 * @throws std::invalid_argument Xとyの長さが異なる場合
 */
inline residual_diagnostics compute_residual_diagnostics(
    const multiple_regression_result& model,
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y)
{
    std::size_t n = X.size();
    if (n != y.size()) {
        throw std::invalid_argument("statcpp::compute_residual_diagnostics: X and y must have same length");
    }

    std::size_t p = X[0].size();
    std::size_t p_full = p + 1;

    // 設計行列を作成
    std::vector<std::vector<double>> X_design(n, std::vector<double>(p_full));
    for (std::size_t i = 0; i < n; ++i) {
        X_design[i][0] = 1.0;
        for (std::size_t j = 0; j < p; ++j) {
            X_design[i][j + 1] = X[i][j];
        }
    }

    // (X^T X)^{-1} を計算
    auto Xt = detail::transpose(X_design);
    auto XtX = detail::matrix_multiply(Xt, X_design);
    auto L = detail::cholesky(XtX);
    auto XtX_inv = detail::inverse_cholesky(L);

    // H = X(X^T X)^{-1}X^T のてこ比（対角要素のみ）
    std::vector<double> hat_values(n);
    for (std::size_t i = 0; i < n; ++i) {
        double h_ii = 0.0;
        for (std::size_t j = 0; j < p_full; ++j) {
            for (std::size_t k = 0; k < p_full; ++k) {
                h_ii += X_design[i][j] * XtX_inv[j][k] * X_design[i][k];
            }
        }
        hat_values[i] = h_ii;
    }

    // 残差を計算
    std::vector<double> residuals(n);
    for (std::size_t i = 0; i < n; ++i) {
        double pred = predict(model, X[i]);
        residuals[i] = y[i] - pred;
    }

    // 標準化残差、スチューデント化残差、クックの距離
    std::vector<double> standardized_residuals(n);
    std::vector<double> studentized_residuals(n);
    std::vector<double> cooks_distance(n);

    double p_d = static_cast<double>(p_full);

    for (std::size_t i = 0; i < n; ++i) {
        standardized_residuals[i] = residuals[i] / model.residual_se;

        double h_i = hat_values[i];
        double se_i = model.residual_se * std::sqrt(1.0 - h_i);
        studentized_residuals[i] = (se_i > 0.0) ? residuals[i] / se_i : 0.0;

        cooks_distance[i] = (standardized_residuals[i] * standardized_residuals[i] / p_d)
                          * (h_i / (1.0 - h_i));
    }

    // ダービン・ワトソン統計量
    double dw_num = 0.0;
    double dw_den = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        dw_den += residuals[i] * residuals[i];
        if (i > 0) {
            double diff = residuals[i] - residuals[i - 1];
            dw_num += diff * diff;
        }
    }
    double durbin_watson = (dw_den > 0.0) ? dw_num / dw_den : 0.0;

    return {residuals, standardized_residuals, studentized_residuals,
            hat_values, cooks_distance, durbin_watson};
}

// ============================================================================
// VIF (Variance Inflation Factor / 分散膨張係数)
// ============================================================================

/**
 * @brief 各説明変数のVIF（分散膨張係数）を計算する
 *
 * VIFは多重共線性の指標で、各説明変数が他の説明変数によって
 * どの程度説明されるかを示します。一般的に VIF > 10 は
 * 多重共線性の問題を示唆します。
 *
 * @param X 説明変数の行列 (n x p)
 * @return std::vector<double> 各説明変数のVIF
 * @throws std::invalid_argument 観測数が3未満の場合
 * @throws std::invalid_argument 説明変数が2未満の場合
 */
inline std::vector<double> compute_vif(const std::vector<std::vector<double>>& X)
{
    std::size_t n = X.size();
    if (n < 3) {
        throw std::invalid_argument("statcpp::compute_vif: need at least 3 observations");
    }

    std::size_t p = X[0].size();
    if (p < 2) {
        throw std::invalid_argument("statcpp::compute_vif: need at least 2 predictors");
    }

    std::vector<double> vif(p);

    for (std::size_t j = 0; j < p; ++j) {
        // j番目の変数を目的変数、他を説明変数として回帰
        std::vector<double> y_j(n);
        std::vector<std::vector<double>> X_others(n, std::vector<double>(p - 1));

        for (std::size_t i = 0; i < n; ++i) {
            y_j[i] = X[i][j];
            std::size_t col = 0;
            for (std::size_t k = 0; k < p; ++k) {
                if (k != j) {
                    X_others[i][col++] = X[i][k];
                }
            }
        }

        auto result = multiple_linear_regression(X_others, y_j);
        double r_sq = result.r_squared;

        // VIF = 1 / (1 - R^2)
        if (r_sq >= 1.0) {
            vif[j] = std::numeric_limits<double>::infinity();
        } else {
            vif[j] = 1.0 / (1.0 - r_sq);
        }
    }

    return vif;
}

// ============================================================================
// Multicollinearity Diagnostics (Extended)
// ============================================================================

/**
 * @brief 相関行列の行列式を計算する
 *
 * 多重共線性の診断に使用します。
 * det(R) が 0 に近いほど多重共線性が強く、
 * det(R) = 1 のとき変数間に相関がないことを示します。
 *
 * @param X 説明変数の行列 (n x p)
 * @return double 相関行列の行列式
 * @throws std::invalid_argument 観測数が2未満の場合
 * @throws std::invalid_argument 説明変数が2未満または4以上の場合（2または3のみサポート）
 */
inline double correlation_matrix_determinant(const std::vector<std::vector<double>>& X)
{
    std::size_t n = X.size();
    if (n < 2) {
        throw std::invalid_argument("statcpp::correlation_matrix_determinant: need at least 2 observations");
    }

    std::size_t p = X[0].size();
    if (p < 2) {
        throw std::invalid_argument("statcpp::correlation_matrix_determinant: need at least 2 predictors");
    }

    // 相関行列を計算
    std::vector<std::vector<double>> corr_matrix(p, std::vector<double>(p));
    for (std::size_t i = 0; i < p; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            if (i == j) {
                corr_matrix[i][j] = 1.0;
            } else if (j > i) {
                // 各列を抽出
                std::vector<double> col_i(n), col_j(n);
                for (std::size_t k = 0; k < n; ++k) {
                    col_i[k] = X[k][i];
                    col_j[k] = X[k][j];
                }
                double corr = pearson_correlation(col_i.begin(), col_i.end(),
                                                   col_j.begin(), col_j.end());
                corr_matrix[i][j] = corr;
                corr_matrix[j][i] = corr;
            }
        }
    }

    // 行列式を計算（小さい行列のみサポート）
    if (p == 2) {
        // 2x2行列: det = a11*a22 - a12*a21
        return corr_matrix[0][0] * corr_matrix[1][1] - corr_matrix[0][1] * corr_matrix[1][0];
    } else if (p == 3) {
        // 3x3行列: サラスの公式
        double a = corr_matrix[0][0] * corr_matrix[1][1] * corr_matrix[2][2];
        double b = corr_matrix[0][1] * corr_matrix[1][2] * corr_matrix[2][0];
        double c = corr_matrix[0][2] * corr_matrix[1][0] * corr_matrix[2][1];
        double d = corr_matrix[0][2] * corr_matrix[1][1] * corr_matrix[2][0];
        double e = corr_matrix[0][0] * corr_matrix[1][2] * corr_matrix[2][1];
        double f = corr_matrix[0][1] * corr_matrix[1][0] * corr_matrix[2][2];
        return a + b + c - d - e - f;
    } else {
        // より大きい行列の場合はLU分解などが必要だが、ここでは非対応
        throw std::invalid_argument("statcpp::correlation_matrix_determinant: only 2 or 3 predictors supported");
    }
}

/**
 * @brief 多重共線性スコアを計算する
 *
 * 0から1の範囲で、1に近いほど多重共線性が強いことを示します。
 * Score = 1 - |det(R)| として計算されます。
 *
 * @param X 説明変数の行列 (n x p)
 * @return double 多重共線性スコア（0: 相関なし、1: 完全な多重共線性）
 * @throws std::invalid_argument 観測数が2未満の場合
 * @throws std::invalid_argument 説明変数が2未満または4以上の場合
 */
inline double multicollinearity_score(const std::vector<std::vector<double>>& X)
{
    double det = correlation_matrix_determinant(X);
    // 行列式は負になることもあるが、0に近いかどうかが重要
    return 1.0 - std::abs(det);
}

// ============================================================================
// R-squared and Related Measures
// ============================================================================

/**
 * @brief 実測値と予測値から決定係数を計算する
 *
 * R^2 = 1 - SS_residual / SS_total として計算されます。
 *
 * @tparam IteratorY 実測値のイテレータ型
 * @tparam IteratorPred 予測値のイテレータ型
 * @param y_first 実測値の開始イテレータ
 * @param y_last 実測値の終了イテレータ
 * @param pred_first 予測値の開始イテレータ
 * @param pred_last 予測値の終了イテレータ
 * @return double 決定係数 R^2
 * @throws std::invalid_argument yと予測値の長さが異なる場合
 * @throws std::invalid_argument 観測数が2未満の場合
 */
template <typename IteratorY, typename IteratorPred>
double r_squared(IteratorY y_first, IteratorY y_last,
                  IteratorPred pred_first, IteratorPred pred_last)
{
    auto n_y = statcpp::count(y_first, y_last);
    auto n_pred = statcpp::count(pred_first, pred_last);

    if (n_y != n_pred) {
        throw std::invalid_argument("statcpp::r_squared: y and predictions must have same length");
    }
    if (n_y < 2) {
        throw std::invalid_argument("statcpp::r_squared: need at least 2 observations");
    }

    double mean_y = statcpp::mean(y_first, y_last);

    double ss_total = 0.0;
    double ss_residual = 0.0;

    auto it_y = y_first;
    auto it_pred = pred_first;
    for (; it_y != y_last; ++it_y, ++it_pred) {
        double y_i = static_cast<double>(*it_y);
        double pred_i = static_cast<double>(*it_pred);
        ss_total += (y_i - mean_y) * (y_i - mean_y);
        ss_residual += (y_i - pred_i) * (y_i - pred_i);
    }

    if (ss_total == 0.0) {
        return 1.0;  // 完全に定数の場合
    }

    return 1.0 - ss_residual / ss_total;
}

/**
 * @brief 自由度調整済み決定係数を計算する
 *
 * 説明変数の数による過剰適合を補正した決定係数です。
 * Adjusted R^2 = 1 - (1 - R^2) * (n - 1) / (n - p - 1)
 *
 * @tparam IteratorY 実測値のイテレータ型
 * @tparam IteratorPred 予測値のイテレータ型
 * @param y_first 実測値の開始イテレータ
 * @param y_last 実測値の終了イテレータ
 * @param pred_first 予測値の開始イテレータ
 * @param pred_last 予測値の終了イテレータ
 * @param num_predictors 説明変数の数（切片を除く）
 * @return double 自由度調整済み決定係数
 * @throws std::invalid_argument yと予測値の長さが異なる場合
 * @throws std::invalid_argument 観測数が説明変数の数+1以下の場合
 */
template <typename IteratorY, typename IteratorPred>
double adjusted_r_squared(IteratorY y_first, IteratorY y_last,
                           IteratorPred pred_first, IteratorPred pred_last,
                           std::size_t num_predictors)
{
    auto n = statcpp::count(y_first, y_last);
    double n_d = static_cast<double>(n);
    double p = static_cast<double>(num_predictors);

    if (n_d <= p + 1.0) {
        throw std::invalid_argument("statcpp::adjusted_r_squared: need more observations than predictors");
    }

    double r_sq = r_squared(y_first, y_last, pred_first, pred_last);
    return 1.0 - (1.0 - r_sq) * (n_d - 1.0) / (n_d - p - 1.0);
}

} // namespace statcpp
