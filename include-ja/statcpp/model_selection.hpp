/**
 * @file model_selection.hpp
 * @brief モデル選択関数
 *
 * AIC、BIC、交差検証、正則化回帰などのモデル選択・評価指標を提供します。
 */

#pragma once

#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"
#include "statcpp/linear_regression.hpp"
#include "statcpp/random_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace statcpp {

// ============================================================================
// Model Selection Criteria (モデル選択基準)
// ============================================================================

/**
 * @brief AIC (Akaike Information Criterion) を計算する
 *
 * 赤池情報量規準を計算します。値が小さいほどモデルが良いことを示します。
 *
 * @param log_likelihood 対数尤度
 * @param k パラメータ数
 * @return AIC値
 */
inline double aic(double log_likelihood, std::size_t k)
{
    return -2.0 * log_likelihood + 2.0 * static_cast<double>(k);
}

/**
 * @brief 単回帰モデルからAICを計算する
 *
 * 単回帰結果から対数尤度を計算し、AICを求めます。
 *
 * @param model 単回帰の結果
 * @param n サンプルサイズ
 * @return AIC値
 */
inline double aic_linear(const simple_regression_result& model, std::size_t n)
{
    // σ² = SS_res / n（MLEバージョン）
    double sigma2 = model.ss_residual / static_cast<double>(n);
    double n_d = static_cast<double>(n);

    // 対数尤度: -n/2 * (log(2π) + log(σ²) + 1)
    double ll = -0.5 * n_d * (std::log(2.0 * pi) + std::log(sigma2) + 1.0);

    return aic(ll, 3);  // k = 2 (係数) + 1 (σ²)
}

/**
 * @brief 重回帰モデルからAICを計算する
 *
 * 重回帰結果から対数尤度を計算し、AICを求めます。
 *
 * @param model 重回帰の結果
 * @param n サンプルサイズ
 * @return AIC値
 */
inline double aic_linear(const multiple_regression_result& model, std::size_t n)
{
    double sigma2 = model.ss_residual / static_cast<double>(n);
    double n_d = static_cast<double>(n);

    double ll = -0.5 * n_d * (std::log(2.0 * pi) + std::log(sigma2) + 1.0);

    std::size_t k = model.coefficients.size() + 1;  // 係数 + σ²
    return aic(ll, k);
}

/**
 * @brief AICc (補正AIC) を計算する
 *
 * 小標本向けに補正されたAICを計算します。
 * サンプルサイズがパラメータ数に対して小さい場合に使用します。
 *
 * @param log_likelihood 対数尤度
 * @param n サンプルサイズ
 * @param k パラメータ数
 * @return AICc値
 * @throws std::invalid_argument n <= k + 1 の場合
 */
inline double aicc(double log_likelihood, std::size_t n, std::size_t k)
{
    double n_d = static_cast<double>(n);
    double k_d = static_cast<double>(k);

    if (n_d <= k_d + 1.0) {
        throw std::invalid_argument("statcpp::aicc: n must be greater than k + 1");
    }

    return aic(log_likelihood, k) + (2.0 * k_d * (k_d + 1.0)) / (n_d - k_d - 1.0);
}

/**
 * @brief BIC (Bayesian Information Criterion) を計算する
 *
 * ベイズ情報量規準（シュワルツ規準）を計算します。
 * AICよりも複雑なモデルに対してペナルティが大きくなります。
 *
 * @param log_likelihood 対数尤度
 * @param n サンプルサイズ
 * @param k パラメータ数
 * @return BIC値
 */
inline double bic(double log_likelihood, std::size_t n, std::size_t k)
{
    return -2.0 * log_likelihood + static_cast<double>(k) * std::log(static_cast<double>(n));
}

/**
 * @brief 単回帰モデルからBICを計算する
 *
 * 単回帰結果から対数尤度を計算し、BICを求めます。
 *
 * @param model 単回帰の結果
 * @param n サンプルサイズ
 * @return BIC値
 */
inline double bic_linear(const simple_regression_result& model, std::size_t n)
{
    double sigma2 = model.ss_residual / static_cast<double>(n);
    double n_d = static_cast<double>(n);

    double ll = -0.5 * n_d * (std::log(2.0 * pi) + std::log(sigma2) + 1.0);

    return bic(ll, n, 3);
}

/**
 * @brief 重回帰モデルからBICを計算する
 *
 * 重回帰結果から対数尤度を計算し、BICを求めます。
 *
 * @param model 重回帰の結果
 * @param n サンプルサイズ
 * @return BIC値
 */
inline double bic_linear(const multiple_regression_result& model, std::size_t n)
{
    double sigma2 = model.ss_residual / static_cast<double>(n);
    double n_d = static_cast<double>(n);

    double ll = -0.5 * n_d * (std::log(2.0 * pi) + std::log(sigma2) + 1.0);

    std::size_t k = model.coefficients.size() + 1;
    return bic(ll, n, k);
}

/**
 * @brief PRESS統計量を計算する
 *
 * Prediction Sum of Squares（予測残差平方和）を計算します。
 * Leave-one-out交差検証の効率的な計算に使用されます。
 *
 * @tparam IteratorX 説明変数のイテレータ型
 * @tparam IteratorY 目的変数のイテレータ型
 * @param x_first 説明変数の開始イテレータ
 * @param x_last 説明変数の終了イテレータ
 * @param y_first 目的変数の開始イテレータ
 * @param y_last 目的変数の終了イテレータ
 * @param model 単回帰モデル
 * @return PRESS統計量
 * @throws std::invalid_argument xとyの長さが異なる場合
 */
template <typename IteratorX, typename IteratorY>
double press_statistic(IteratorX x_first, IteratorX x_last,
                        IteratorY y_first, IteratorY y_last,
                        const simple_regression_result& model)
{
    auto n = statcpp::count(x_first, x_last);
    auto n_y = statcpp::count(y_first, y_last);
    if (n != n_y) {
        throw std::invalid_argument("statcpp::press_statistic: x and y must have same length");
    }

    double n_d = static_cast<double>(n);
    double mean_x = statcpp::mean(x_first, x_last);

    // Sxx を計算
    double sxx = 0.0;
    for (auto it = x_first; it != x_last; ++it) {
        double dx = static_cast<double>(*it) - mean_x;
        sxx += dx * dx;
    }

    double press = 0.0;
    auto it_x = x_first;
    auto it_y = y_first;
    for (; it_x != x_last; ++it_x, ++it_y) {
        double x_i = static_cast<double>(*it_x);
        double y_i = static_cast<double>(*it_y);

        double y_hat = predict(model, x_i);
        double residual = y_i - y_hat;

        // てこ比 h_ii
        double dx = x_i - mean_x;
        double h_ii = 1.0 / n_d + dx * dx / sxx;

        // PRESS残差 = 残差 / (1 - h_ii)
        double press_residual = residual / (1.0 - h_ii);
        press += press_residual * press_residual;
    }

    return press;
}

// ============================================================================
// Cross-Validation (交差検証)
// ============================================================================

/**
 * @brief 交差検証の結果を格納する構造体
 */
struct cv_result {
    double mean_error;              /**< 平均誤差（MSE, MAE等） */
    double se_error;                /**< 誤差の標準誤差 */
    std::vector<double> fold_errors; /**< 各フォールドの誤差 */
    std::size_t n_folds;            /**< フォールド数 */
};

/**
 * @brief k-fold交差検証用のインデックスを生成する
 *
 * データをk個のフォールドに分割するためのインデックスを生成します。
 *
 * @param n データサイズ
 * @param k フォールド数
 * @param shuffle シャッフルするかどうか（デフォルト: true）
 * @return 各フォールドに属するインデックスのベクタ
 * @throws std::invalid_argument kが2未満の場合、またはkがnを超える場合
 */
inline std::vector<std::vector<std::size_t>> create_cv_folds(
    std::size_t n, std::size_t k, bool shuffle = true)
{
    if (k < 2) {
        throw std::invalid_argument("statcpp::create_cv_folds: k must be at least 2");
    }
    if (k > n) {
        throw std::invalid_argument("statcpp::create_cv_folds: k cannot exceed n");
    }

    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    if (shuffle) {
        std::shuffle(indices.begin(), indices.end(), get_random_engine());
    }

    std::vector<std::vector<std::size_t>> folds(k);
    std::size_t fold_size = n / k;
    std::size_t remainder = n % k;

    std::size_t current = 0;
    for (std::size_t i = 0; i < k; ++i) {
        std::size_t this_fold_size = fold_size + (i < remainder ? 1 : 0);
        for (std::size_t j = 0; j < this_fold_size; ++j) {
            folds[i].push_back(indices[current++]);
        }
    }

    return folds;
}

/**
 * @brief 重回帰モデルのk-fold交差検証を実行する
 *
 * 指定したフォールド数でデータを分割し、交差検証によりモデルの予測性能を評価します。
 *
 * @param X 説明変数行列（各行が1サンプル）
 * @param y 目的変数ベクタ
 * @param k フォールド数（デフォルト: 5）
 * @return 交差検証の結果
 * @throws std::invalid_argument Xとyのサイズが異なる場合
 */
inline cv_result cross_validate_linear(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    std::size_t k = 5)
{
    std::size_t n = X.size();
    if (n != y.size()) {
        throw std::invalid_argument("statcpp::cross_validate_linear: X and y must have same size");
    }

    auto folds = create_cv_folds(n, k, true);
    std::vector<double> fold_errors(k);

    for (std::size_t fold = 0; fold < k; ++fold) {
        // テストセットとトレーニングセットを分離
        std::vector<std::size_t> test_idx = folds[fold];
        std::vector<std::size_t> train_idx;
        for (std::size_t f = 0; f < k; ++f) {
            if (f != fold) {
                train_idx.insert(train_idx.end(), folds[f].begin(), folds[f].end());
            }
        }

        // トレーニングデータ
        std::vector<std::vector<double>> X_train(train_idx.size());
        std::vector<double> y_train(train_idx.size());
        for (std::size_t i = 0; i < train_idx.size(); ++i) {
            X_train[i] = X[train_idx[i]];
            y_train[i] = y[train_idx[i]];
        }

        // テストデータ
        std::vector<std::vector<double>> X_test(test_idx.size());
        std::vector<double> y_test(test_idx.size());
        for (std::size_t i = 0; i < test_idx.size(); ++i) {
            X_test[i] = X[test_idx[i]];
            y_test[i] = y[test_idx[i]];
        }

        // モデルをフィット
        try {
            auto model = multiple_linear_regression(X_train, y_train);

            // テスト誤差を計算
            double mse = 0.0;
            for (std::size_t i = 0; i < test_idx.size(); ++i) {
                double pred = predict(model, X_test[i]);
                double err = y_test[i] - pred;
                mse += err * err;
            }
            fold_errors[fold] = mse / static_cast<double>(test_idx.size());
        } catch (...) {
            fold_errors[fold] = std::numeric_limits<double>::infinity();
        }
    }

    double mean_error = statcpp::mean(fold_errors.begin(), fold_errors.end());
    double se_error = statcpp::sample_stddev(fold_errors.begin(), fold_errors.end())
                    / std::sqrt(static_cast<double>(k));

    return {mean_error, se_error, fold_errors, k};
}

/**
 * @brief Leave-one-out交差検証を実行する
 *
 * 各サンプルを1つずつテストデータとして使用する交差検証を実行します。
 *
 * @param X 説明変数行列（各行が1サンプル）
 * @param y 目的変数ベクタ
 * @return 交差検証の結果
 */
inline cv_result loocv_linear(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y)
{
    return cross_validate_linear(X, y, X.size());
}

// ============================================================================
// Regularized Regression (正則化回帰)
// ============================================================================

/**
 * @brief 正則化回帰の結果を格納する構造体
 */
struct regularized_regression_result {
    std::vector<double> coefficients;   /**< 回帰係数（切片を含む） */
    double lambda;                      /**< 正則化パラメータ */
    double mse;                         /**< 平均二乗誤差 */
    std::size_t iterations;             /**< 反復回数 */
    bool converged;                     /**< 収束フラグ */
};

/**
 * @brief Ridge回帰（L2正則化）を実行する
 *
 * 座標降下法によりRidge回帰を解きます。
 * L2ペナルティにより係数を縮小し、多重共線性に対処します。
 *
 * @param X 説明変数行列（各行が1サンプル、切片列を含まない）
 * @param y 目的変数ベクタ
 * @param lambda 正則化パラメータ（>= 0）
 * @param standardize データを標準化するかどうか（デフォルト: true）
 * @param max_iter 最大反復回数（デフォルト: 1000）
 * @param tol 収束判定の許容誤差（デフォルト: 1e-6）
 * @return 正則化回帰の結果
 * @throws std::invalid_argument lambdaが負の場合、データが空の場合、Xとyのサイズが異なる場合
 */
inline regularized_regression_result ridge_regression(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    double lambda,
    bool standardize = true,
    std::size_t max_iter = 1000,
    double tol = 1e-6)
{
    // X に切片列が含まれていないことを確認（linear_regression.hppから）
    statcpp::detail::validate_no_intercept_column(X, "ridge_regression");

    if (lambda < 0.0) {
        throw std::invalid_argument("statcpp::ridge_regression: lambda must be non-negative");
    }

    std::size_t n = X.size();
    if (n == 0) {
        throw std::invalid_argument("statcpp::ridge_regression: empty data");
    }
    if (n != y.size()) {
        throw std::invalid_argument("statcpp::ridge_regression: X and y must have same size");
    }

    std::size_t p = X[0].size();

    // データの標準化
    std::vector<double> X_mean(p, 0.0);
    std::vector<double> X_std(p, 1.0);
    double y_mean = statcpp::mean(y.begin(), y.end());

    std::vector<std::vector<double>> X_scaled = X;
    std::vector<double> y_centered(n);

    if (standardize) {
        for (std::size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                sum += X[i][j];
            }
            X_mean[j] = sum / static_cast<double>(n);

            double ss = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                double d = X[i][j] - X_mean[j];
                ss += d * d;
            }
            X_std[j] = std::sqrt(ss / static_cast<double>(n));
            if (X_std[j] < 1e-10) X_std[j] = 1.0;

            for (std::size_t i = 0; i < n; ++i) {
                X_scaled[i][j] = (X[i][j] - X_mean[j]) / X_std[j];
            }
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        y_centered[i] = y[i] - y_mean;
    }

    // Ridge閉形式解: β = (X'X + λI)^{-1} X'y
    // 座標降下法で解く
    std::vector<double> beta(p, 0.0);
    std::vector<double> residuals = y_centered;

    std::size_t iter = 0;
    bool converged = false;

    for (iter = 0; iter < max_iter; ++iter) {
        double max_change = 0.0;

        for (std::size_t j = 0; j < p; ++j) {
            // 現在の残差にこの変数の寄与を足し戻す
            for (std::size_t i = 0; i < n; ++i) {
                residuals[i] += X_scaled[i][j] * beta[j];
            }

            // X_j'r を計算
            double xr = 0.0;
            double xx = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                xr += X_scaled[i][j] * residuals[i];
                xx += X_scaled[i][j] * X_scaled[i][j];
            }

            // Ridge更新
            double beta_new = xr / (xx + lambda);
            double change = std::abs(beta_new - beta[j]);
            max_change = std::max(max_change, change);

            beta[j] = beta_new;

            // 残差を更新
            for (std::size_t i = 0; i < n; ++i) {
                residuals[i] -= X_scaled[i][j] * beta[j];
            }
        }

        if (max_change < tol) {
            converged = true;
            ++iter;
            break;
        }
    }

    // 係数を元のスケールに戻す
    std::vector<double> coefficients(p + 1);
    if (standardize) {
        coefficients[0] = y_mean;
        for (std::size_t j = 0; j < p; ++j) {
            coefficients[j + 1] = beta[j] / X_std[j];
            coefficients[0] -= coefficients[j + 1] * X_mean[j];
        }
    } else {
        coefficients[0] = y_mean;
        for (std::size_t j = 0; j < p; ++j) {
            coefficients[j + 1] = beta[j];
        }
    }

    // MSEを計算
    double mse = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double pred = coefficients[0];
        for (std::size_t j = 0; j < p; ++j) {
            pred += coefficients[j + 1] * X[i][j];
        }
        double err = y[i] - pred;
        mse += err * err;
    }
    mse /= static_cast<double>(n);

    return {coefficients, lambda, mse, iter, converged};
}

/**
 * @brief Lasso回帰（L1正則化）を実行する
 *
 * 座標降下法によりLasso回帰を解きます。
 * L1ペナルティにより一部の係数を完全に0にし、変数選択を行います。
 *
 * @param X 説明変数行列（各行が1サンプル、切片列を含まない）
 * @param y 目的変数ベクタ
 * @param lambda 正則化パラメータ（>= 0）
 * @param standardize データを標準化するかどうか（デフォルト: true）
 * @param max_iter 最大反復回数（デフォルト: 1000）
 * @param tol 収束判定の許容誤差（デフォルト: 1e-6）
 * @return 正則化回帰の結果
 * @throws std::invalid_argument lambdaが負の場合、データが空の場合、Xとyのサイズが異なる場合
 */
inline regularized_regression_result lasso_regression(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    double lambda,
    bool standardize = true,
    std::size_t max_iter = 1000,
    double tol = 1e-6)
{
    // X に切片列が含まれていないことを確認
    statcpp::detail::validate_no_intercept_column(X, "lasso_regression");

    if (lambda < 0.0) {
        throw std::invalid_argument("statcpp::lasso_regression: lambda must be non-negative");
    }

    std::size_t n = X.size();
    if (n == 0) {
        throw std::invalid_argument("statcpp::lasso_regression: empty data");
    }
    if (n != y.size()) {
        throw std::invalid_argument("statcpp::lasso_regression: X and y must have same size");
    }

    std::size_t p = X[0].size();

    // データの標準化
    std::vector<double> X_mean(p, 0.0);
    std::vector<double> X_std(p, 1.0);
    double y_mean = statcpp::mean(y.begin(), y.end());

    std::vector<std::vector<double>> X_scaled = X;
    std::vector<double> y_centered(n);

    if (standardize) {
        for (std::size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                sum += X[i][j];
            }
            X_mean[j] = sum / static_cast<double>(n);

            double ss = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                double d = X[i][j] - X_mean[j];
                ss += d * d;
            }
            X_std[j] = std::sqrt(ss / static_cast<double>(n));
            if (X_std[j] < 1e-10) X_std[j] = 1.0;

            for (std::size_t i = 0; i < n; ++i) {
                X_scaled[i][j] = (X[i][j] - X_mean[j]) / X_std[j];
            }
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        y_centered[i] = y[i] - y_mean;
    }

    // 座標降下法
    std::vector<double> beta(p, 0.0);
    std::vector<double> residuals = y_centered;

    // Soft thresholding関数
    auto soft_threshold = [](double x, double t) -> double {
        if (x > t) return x - t;
        if (x < -t) return x + t;
        return 0.0;
    };

    std::size_t iter = 0;
    bool converged = false;

    for (iter = 0; iter < max_iter; ++iter) {
        double max_change = 0.0;

        for (std::size_t j = 0; j < p; ++j) {
            // 現在の残差にこの変数の寄与を足し戻す
            for (std::size_t i = 0; i < n; ++i) {
                residuals[i] += X_scaled[i][j] * beta[j];
            }

            // X_j'r を計算
            double xr = 0.0;
            double xx = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                xr += X_scaled[i][j] * residuals[i];
                xx += X_scaled[i][j] * X_scaled[i][j];
            }

            // Lasso更新（soft thresholding）
            double beta_new = soft_threshold(xr, lambda) / xx;
            double change = std::abs(beta_new - beta[j]);
            max_change = std::max(max_change, change);

            beta[j] = beta_new;

            // 残差を更新
            for (std::size_t i = 0; i < n; ++i) {
                residuals[i] -= X_scaled[i][j] * beta[j];
            }
        }

        if (max_change < tol) {
            converged = true;
            ++iter;
            break;
        }
    }

    // 係数を元のスケールに戻す
    std::vector<double> coefficients(p + 1);
    if (standardize) {
        coefficients[0] = y_mean;
        for (std::size_t j = 0; j < p; ++j) {
            coefficients[j + 1] = beta[j] / X_std[j];
            coefficients[0] -= coefficients[j + 1] * X_mean[j];
        }
    } else {
        coefficients[0] = y_mean;
        for (std::size_t j = 0; j < p; ++j) {
            coefficients[j + 1] = beta[j];
        }
    }

    // MSEを計算
    double mse = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double pred = coefficients[0];
        for (std::size_t j = 0; j < p; ++j) {
            pred += coefficients[j + 1] * X[i][j];
        }
        double err = y[i] - pred;
        mse += err * err;
    }
    mse /= static_cast<double>(n);

    return {coefficients, lambda, mse, iter, converged};
}

/**
 * @brief Elastic Net回帰（L1 + L2正則化）を実行する
 *
 * 座標降下法によりElastic Net回帰を解きます。
 * L1とL2ペナルティを組み合わせ、Lassoの変数選択とRidgeの安定性を両立します。
 *
 * @param X 説明変数行列（各行が1サンプル、切片列を含まない）
 * @param y 目的変数ベクタ
 * @param lambda 正則化パラメータ（>= 0）
 * @param alpha L1ペナルティの比率（0 = Ridge, 1 = Lasso、デフォルト: 0.5）
 * @param standardize データを標準化するかどうか（デフォルト: true）
 * @param max_iter 最大反復回数（デフォルト: 1000）
 * @param tol 収束判定の許容誤差（デフォルト: 1e-6）
 * @return 正則化回帰の結果
 * @throws std::invalid_argument lambdaが負の場合、alphaが[0,1]の範囲外の場合、データが空の場合、Xとyのサイズが異なる場合
 */
inline regularized_regression_result elastic_net_regression(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    double lambda,
    double alpha = 0.5,  // L1の比率 (0 = Ridge, 1 = Lasso)
    bool standardize = true,
    std::size_t max_iter = 1000,
    double tol = 1e-6)
{
    // X に切片列が含まれていないことを確認
    statcpp::detail::validate_no_intercept_column(X, "elastic_net_regression");

    if (lambda < 0.0) {
        throw std::invalid_argument("statcpp::elastic_net_regression: lambda must be non-negative");
    }
    if (alpha < 0.0 || alpha > 1.0) {
        throw std::invalid_argument("statcpp::elastic_net_regression: alpha must be in [0, 1]");
    }

    std::size_t n = X.size();
    if (n == 0) {
        throw std::invalid_argument("statcpp::elastic_net_regression: empty data");
    }
    if (n != y.size()) {
        throw std::invalid_argument("statcpp::elastic_net_regression: X and y must have same size");
    }

    std::size_t p = X[0].size();

    // データの標準化
    std::vector<double> X_mean(p, 0.0);
    std::vector<double> X_std(p, 1.0);
    double y_mean = statcpp::mean(y.begin(), y.end());

    std::vector<std::vector<double>> X_scaled = X;
    std::vector<double> y_centered(n);

    if (standardize) {
        for (std::size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                sum += X[i][j];
            }
            X_mean[j] = sum / static_cast<double>(n);

            double ss = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                double d = X[i][j] - X_mean[j];
                ss += d * d;
            }
            X_std[j] = std::sqrt(ss / static_cast<double>(n));
            if (X_std[j] < 1e-10) X_std[j] = 1.0;

            for (std::size_t i = 0; i < n; ++i) {
                X_scaled[i][j] = (X[i][j] - X_mean[j]) / X_std[j];
            }
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        y_centered[i] = y[i] - y_mean;
    }

    // 座標降下法
    std::vector<double> beta(p, 0.0);
    std::vector<double> residuals = y_centered;

    auto soft_threshold = [](double x, double t) -> double {
        if (x > t) return x - t;
        if (x < -t) return x + t;
        return 0.0;
    };

    double lambda1 = alpha * lambda;        // L1ペナルティ
    double lambda2 = (1.0 - alpha) * lambda; // L2ペナルティ

    std::size_t iter = 0;
    bool converged = false;

    for (iter = 0; iter < max_iter; ++iter) {
        double max_change = 0.0;

        for (std::size_t j = 0; j < p; ++j) {
            for (std::size_t i = 0; i < n; ++i) {
                residuals[i] += X_scaled[i][j] * beta[j];
            }

            double xr = 0.0;
            double xx = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                xr += X_scaled[i][j] * residuals[i];
                xx += X_scaled[i][j] * X_scaled[i][j];
            }

            // Elastic Net更新
            double beta_new = soft_threshold(xr, lambda1) / (xx + lambda2);
            double change = std::abs(beta_new - beta[j]);
            max_change = std::max(max_change, change);

            beta[j] = beta_new;

            for (std::size_t i = 0; i < n; ++i) {
                residuals[i] -= X_scaled[i][j] * beta[j];
            }
        }

        if (max_change < tol) {
            converged = true;
            ++iter;
            break;
        }
    }

    // 係数を元のスケールに戻す
    std::vector<double> coefficients(p + 1);
    if (standardize) {
        coefficients[0] = y_mean;
        for (std::size_t j = 0; j < p; ++j) {
            coefficients[j + 1] = beta[j] / X_std[j];
            coefficients[0] -= coefficients[j + 1] * X_mean[j];
        }
    } else {
        coefficients[0] = y_mean;
        for (std::size_t j = 0; j < p; ++j) {
            coefficients[j + 1] = beta[j];
        }
    }

    // MSEを計算
    double mse = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double pred = coefficients[0];
        for (std::size_t j = 0; j < p; ++j) {
            pred += coefficients[j + 1] * X[i][j];
        }
        double err = y[i] - pred;
        mse += err * err;
    }
    mse /= static_cast<double>(n);

    return {coefficients, lambda, mse, iter, converged};
}

// ============================================================================
// Lambda Selection (正則化パラメータの選択)
// ============================================================================

/**
 * @brief 交差検証によりRidge回帰の最適なlambdaを選択する
 *
 * 指定したlambdaのグリッドに対してk-fold交差検証を行い、
 * 最小の交差検証誤差を与えるlambdaを選択します。
 *
 * @param X 説明変数行列（各行が1サンプル）
 * @param y 目的変数ベクタ
 * @param lambda_grid 評価するlambda値のベクタ
 * @param k フォールド数（デフォルト: 5）
 * @return 最適なlambdaと各lambdaに対する交差検証誤差のペア
 */
inline std::pair<double, std::vector<double>> cv_ridge(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    const std::vector<double>& lambda_grid,
    std::size_t k = 5)
{
    std::vector<double> cv_errors(lambda_grid.size());

    for (std::size_t l = 0; l < lambda_grid.size(); ++l) {
        double lambda = lambda_grid[l];
        auto folds = create_cv_folds(X.size(), k, true);

        double total_error = 0.0;
        for (std::size_t fold = 0; fold < k; ++fold) {
            std::vector<std::size_t> test_idx = folds[fold];
            std::vector<std::size_t> train_idx;
            for (std::size_t f = 0; f < k; ++f) {
                if (f != fold) {
                    train_idx.insert(train_idx.end(), folds[f].begin(), folds[f].end());
                }
            }

            std::vector<std::vector<double>> X_train(train_idx.size());
            std::vector<double> y_train(train_idx.size());
            for (std::size_t i = 0; i < train_idx.size(); ++i) {
                X_train[i] = X[train_idx[i]];
                y_train[i] = y[train_idx[i]];
            }

            try {
                auto model = ridge_regression(X_train, y_train, lambda);

                double mse = 0.0;
                for (std::size_t i : test_idx) {
                    double pred = model.coefficients[0];
                    for (std::size_t j = 0; j < X[i].size(); ++j) {
                        pred += model.coefficients[j + 1] * X[i][j];
                    }
                    double err = y[i] - pred;
                    mse += err * err;
                }
                total_error += mse / static_cast<double>(test_idx.size());
            } catch (...) {
                total_error += std::numeric_limits<double>::infinity();
            }
        }
        cv_errors[l] = total_error / static_cast<double>(k);
    }

    // 最小誤差のlambdaを選択
    auto min_it = std::min_element(cv_errors.begin(), cv_errors.end());
    double best_lambda = lambda_grid[std::distance(cv_errors.begin(), min_it)];

    return {best_lambda, cv_errors};
}

/**
 * @brief 交差検証によりLasso回帰の最適なlambdaを選択する
 *
 * 指定したlambdaのグリッドに対してk-fold交差検証を行い、
 * 最小の交差検証誤差を与えるlambdaを選択します。
 *
 * @param X 説明変数行列（各行が1サンプル）
 * @param y 目的変数ベクタ
 * @param lambda_grid 評価するlambda値のベクタ
 * @param k フォールド数（デフォルト: 5）
 * @return 最適なlambdaと各lambdaに対する交差検証誤差のペア
 */
inline std::pair<double, std::vector<double>> cv_lasso(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    const std::vector<double>& lambda_grid,
    std::size_t k = 5)
{
    std::vector<double> cv_errors(lambda_grid.size());

    for (std::size_t l = 0; l < lambda_grid.size(); ++l) {
        double lambda = lambda_grid[l];
        auto folds = create_cv_folds(X.size(), k, true);

        double total_error = 0.0;
        for (std::size_t fold = 0; fold < k; ++fold) {
            std::vector<std::size_t> test_idx = folds[fold];
            std::vector<std::size_t> train_idx;
            for (std::size_t f = 0; f < k; ++f) {
                if (f != fold) {
                    train_idx.insert(train_idx.end(), folds[f].begin(), folds[f].end());
                }
            }

            std::vector<std::vector<double>> X_train(train_idx.size());
            std::vector<double> y_train(train_idx.size());
            for (std::size_t i = 0; i < train_idx.size(); ++i) {
                X_train[i] = X[train_idx[i]];
                y_train[i] = y[train_idx[i]];
            }

            try {
                auto model = lasso_regression(X_train, y_train, lambda);

                double mse = 0.0;
                for (std::size_t i : test_idx) {
                    double pred = model.coefficients[0];
                    for (std::size_t j = 0; j < X[i].size(); ++j) {
                        pred += model.coefficients[j + 1] * X[i][j];
                    }
                    double err = y[i] - pred;
                    mse += err * err;
                }
                total_error += mse / static_cast<double>(test_idx.size());
            } catch (...) {
                total_error += std::numeric_limits<double>::infinity();
            }
        }
        cv_errors[l] = total_error / static_cast<double>(k);
    }

    auto min_it = std::min_element(cv_errors.begin(), cv_errors.end());
    double best_lambda = lambda_grid[std::distance(cv_errors.begin(), min_it)];

    return {best_lambda, cv_errors};
}

/**
 * @brief 正則化回帰用のlambdaグリッドを自動生成する
 *
 * データに基づいてlambdaの最大値を計算し、
 * 対数スケールでlambdaのグリッドを生成します。
 *
 * @param X 説明変数行列（各行が1サンプル）
 * @param y 目的変数ベクタ
 * @param n_lambda グリッドのサイズ（デフォルト: 100）
 * @param lambda_min_ratio lambda_maxに対するlambda_minの比率（デフォルト: 0.0001）
 * @return 対数スケールで等間隔に配置されたlambda値のベクタ
 */
inline std::vector<double> generate_lambda_grid(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    std::size_t n_lambda = 100,
    double lambda_min_ratio = 0.0001)
{
    std::size_t n = X.size();
    std::size_t p = X[0].size();

    // y を中心化
    double y_mean = statcpp::mean(y.begin(), y.end());

    // lambda_max を計算（全ての係数が0になるlambda）
    double lambda_max = 0.0;
    for (std::size_t j = 0; j < p; ++j) {
        double xy = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            xy += X[i][j] * (y[i] - y_mean);
        }
        lambda_max = std::max(lambda_max, std::abs(xy) / static_cast<double>(n));
    }

    double lambda_min = lambda_max * lambda_min_ratio;

    // 対数スケールでグリッドを生成
    std::vector<double> grid(n_lambda);
    double log_max = std::log(lambda_max);
    double log_min = std::log(lambda_min);
    double step = (log_max - log_min) / static_cast<double>(n_lambda - 1);

    for (std::size_t i = 0; i < n_lambda; ++i) {
        grid[i] = std::exp(log_max - static_cast<double>(i) * step);
    }

    return grid;
}

} // namespace statcpp
