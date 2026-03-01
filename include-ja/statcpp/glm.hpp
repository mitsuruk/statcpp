/**
 * @file glm.hpp
 * @brief 一般化線形モデル（GLM）
 *
 * ロジスティック回帰、ポアソン回帰などの一般化線形モデルを提供します。
 * IRLS（反復再重み付け最小二乗法）アルゴリズムを使用してパラメータを推定します。
 */

#pragma once

#include "statcpp/basic_statistics.hpp"
#include "statcpp/continuous_distributions.hpp"
#include "statcpp/linear_regression.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace statcpp {

// ============================================================================
// GLM Result Structures
// ============================================================================

/**
 * @brief リンク関数の種類
 *
 * 一般化線形モデルで使用するリンク関数を指定します。
 */
enum class link_function {
    identity,       ///< 恒等リンク（線形回帰）
    logit,          ///< ロジットリンク（ロジスティック回帰）
    probit,         ///< プロビットリンク
    log,            ///< 対数リンク（ポアソン回帰）
    inverse,        ///< 逆数リンク（ガンマ回帰）
    cloglog         ///< 補対数対数リンク
};

/**
 * @brief 分布族
 *
 * 一般化線形モデルで使用する確率分布族を指定します。
 */
enum class distribution_family {
    gaussian,       ///< 正規分布
    binomial,       ///< 二項分布
    poisson,        ///< ポアソン分布
    gamma_family    ///< ガンマ分布（gamma は予約語なので gamma_family）
};

/**
 * @brief GLM結果構造体
 *
 * 一般化線形モデルのフィッティング結果を格納します。
 */
struct glm_result {
    std::vector<double> coefficients;       ///< 回帰係数
    std::vector<double> coefficient_se;     ///< 係数の標準誤差
    std::vector<double> z_statistics;       ///< z統計量（またはWald統計量）
    std::vector<double> p_values;           ///< p値
    double null_deviance;                   ///< ヌル逸脱度
    double residual_deviance;               ///< 残差逸脱度
    double df_null;                         ///< ヌルモデルの自由度
    double df_residual;                     ///< 残差の自由度
    double aic;                             ///< AIC
    double bic;                             ///< BIC
    double log_likelihood;                  ///< 対数尤度
    std::size_t iterations;                 ///< 収束までの反復回数
    bool converged;                         ///< 収束したかどうか
    link_function link;                     ///< 使用したリンク関数
    distribution_family family;             ///< 使用した分布族
};

// ============================================================================
// Link Functions (リンク関数)
// ============================================================================

namespace detail {

/**
 * @brief リンク関数 g(mu) -> eta
 *
 * 期待値 mu をリンク関数で変換して線形予測子 eta を返します。
 *
 * @param mu 期待値
 * @param link 使用するリンク関数
 * @return 線形予測子 eta
 */
inline double link_transform(double mu, link_function link)
{
    switch (link) {
        case link_function::identity:
            return mu;
        case link_function::logit:
            mu = std::max(1e-10, std::min(1.0 - 1e-10, mu));
            return std::log(mu / (1.0 - mu));
        case link_function::probit:
            mu = std::max(1e-10, std::min(1.0 - 1e-10, mu));
            return norm_quantile(mu);
        case link_function::log:
            return std::log(std::max(1e-10, mu));
        case link_function::inverse:
            return 1.0 / std::max(1e-10, mu);
        case link_function::cloglog:
            mu = std::max(1e-10, std::min(1.0 - 1e-10, mu));
            return std::log(-std::log(1.0 - mu));
        default:
            return mu;
    }
}

/**
 * @brief 逆リンク関数 g^{-1}(eta) -> mu
 *
 * 線形予測子 eta を逆リンク関数で変換して期待値 mu を返します。
 *
 * @param eta 線形予測子
 * @param link 使用するリンク関数
 * @return 期待値 mu
 */
inline double inverse_link(double eta, link_function link)
{
    switch (link) {
        case link_function::identity:
            return eta;
        case link_function::logit:
            return 1.0 / (1.0 + std::exp(-eta));
        case link_function::probit:
            return norm_cdf(eta);
        case link_function::log:
            return std::exp(eta);
        case link_function::inverse:
            return 1.0 / eta;
        case link_function::cloglog:
            return 1.0 - std::exp(-std::exp(eta));
        default:
            return eta;
    }
}

/**
 * @brief リンク関数の微分 d(eta)/d(mu) = g'(mu)
 *
 * リンク関数の期待値 mu に関する微分を計算します。
 *
 * @param mu 期待値
 * @param link 使用するリンク関数
 * @return リンク関数の微分値
 * @throws std::runtime_error cloglog リンクで mu が 0 に近い場合
 */
inline double link_derivative(double mu, link_function link)
{
    switch (link) {
        case link_function::identity:
            return 1.0;
        case link_function::logit:
            mu = std::max(1e-10, std::min(1.0 - 1e-10, mu));
            return 1.0 / (mu * (1.0 - mu));
        case link_function::probit:
            mu = std::max(1e-10, std::min(1.0 - 1e-10, mu));
            return 1.0 / normal_pdf(norm_quantile(mu));
        case link_function::log:
            return 1.0 / std::max(1e-10, mu);
        case link_function::inverse:
            mu = std::max(1e-10, mu);
            return -1.0 / (mu * mu);
        case link_function::cloglog: {
            mu = std::max(1e-8, std::min(1.0 - 1e-8, mu));
            double neg_log_term = -std::log(1.0 - mu);  // -log(1-mu) > 0 (0 < mu < 1)
            // -log(1-mu)が0に近い場合の保護（mu が 0 に近い）
            if (neg_log_term < 1e-10) {
                throw std::runtime_error("statcpp::link_derivative: cloglog derivative undefined near mu=0");
            }
            // g'(mu) = 1 / ((1-mu) * (-log(1-mu)))
            return 1.0 / ((1.0 - mu) * neg_log_term);
        }
        default:
            return 1.0;
    }
}

/**
 * @brief 分散関数 V(mu)
 *
 * 分布族に応じた分散関数を計算します。
 *
 * @param mu 期待値
 * @param family 分布族
 * @return 分散関数の値
 */
inline double variance_function(double mu, distribution_family family)
{
    switch (family) {
        case distribution_family::gaussian:
            return 1.0;
        case distribution_family::binomial:
            mu = std::max(1e-10, std::min(1.0 - 1e-10, mu));
            return mu * (1.0 - mu);
        case distribution_family::poisson:
            return std::max(1e-10, mu);
        case distribution_family::gamma_family:
            mu = std::max(1e-10, mu);
            return mu * mu;
        default:
            return 1.0;
    }
}

/**
 * @brief 逸脱度の計算（一つの観測に対して）
 *
 * 分布族に応じた逸脱度残差を計算します。
 *
 * @param y 観測値
 * @param mu 期待値
 * @param family 分布族
 * @return 逸脱度残差
 */
inline double deviance_residual(double y, double mu, distribution_family family)
{
    switch (family) {
        case distribution_family::gaussian:
            return (y - mu) * (y - mu);
        case distribution_family::binomial:
            {
                mu = std::max(1e-10, std::min(1.0 - 1e-10, mu));
                double d = 0.0;
                if (y > 0.0) {
                    d += y * std::log(y / mu);
                }
                if (y < 1.0) {
                    d += (1.0 - y) * std::log((1.0 - y) / (1.0 - mu));
                }
                return 2.0 * d;
            }
        case distribution_family::poisson:
            {
                mu = std::max(1e-10, mu);
                if (y > 0.0) {
                    return 2.0 * (y * std::log(y / mu) - (y - mu));
                } else {
                    return 2.0 * mu;
                }
            }
        case distribution_family::gamma_family:
            {
                mu = std::max(1e-10, mu);
                return 2.0 * ((y - mu) / mu - std::log(y / mu));
            }
        default:
            return (y - mu) * (y - mu);
    }
}

/**
 * @brief 重み付き最小二乗の解を求める
 *
 * (X'WX)^{-1} X'Wz を Cholesky 分解を用いて計算します。
 *
 * @param X 設計行列
 * @param z 作業変数ベクトル
 * @param w 重みベクトル
 * @param XtWX_inv 出力: (X'WX)^{-1} の逆行列
 * @return 重み付き最小二乗解（係数ベクトル）
 * @throws std::runtime_error 行列が正定値でない場合
 */
inline std::vector<double> solve_weighted_least_squares(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& z,
    const std::vector<double>& w,
    std::vector<std::vector<double>>& XtWX_inv)  // 出力: (X'WX)^{-1}
{
    std::size_t n = X.size();
    std::size_t p = X[0].size();

    // X'WX を計算
    std::vector<std::vector<double>> XtWX(p, std::vector<double>(p, 0.0));
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t k = 0; k < p; ++k) {
            for (std::size_t i = 0; i < n; ++i) {
                XtWX[j][k] += X[i][j] * w[i] * X[i][k];
            }
        }
    }

    // X'Wz を計算
    std::vector<double> XtWz(p, 0.0);
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            XtWz[j] += X[i][j] * w[i] * z[i];
        }
    }

    // Cholesky分解で解く
    std::vector<std::vector<double>> L(p, std::vector<double>(p, 0.0));
    for (std::size_t i = 0; i < p; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < j; ++k) {
                sum += L[i][k] * L[j][k];
            }
            if (i == j) {
                double val = XtWX[i][i] - sum;
                if (val <= 0.0) {
                    throw std::runtime_error("statcpp::glm: matrix is not positive definite");
                }
                L[i][j] = std::sqrt(val);
            } else {
                L[i][j] = (XtWX[i][j] - sum) / L[j][j];
            }
        }
    }

    // 前進代入
    std::vector<double> y(p);
    for (std::size_t i = 0; i < p; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < i; ++j) {
            sum += L[i][j] * y[j];
        }
        y[i] = (XtWz[i] - sum) / L[i][i];
    }

    // 後退代入
    std::vector<double> beta(p);
    for (std::size_t i = p; i > 0; --i) {
        std::size_t idx = i - 1;
        double sum = 0.0;
        for (std::size_t j = idx + 1; j < p; ++j) {
            sum += L[j][idx] * beta[j];
        }
        beta[idx] = (y[idx] - sum) / L[idx][idx];
    }

    // (X'WX)^{-1} を計算
    XtWX_inv.assign(p, std::vector<double>(p, 0.0));
    for (std::size_t col = 0; col < p; ++col) {
        std::vector<double> e(p, 0.0);
        e[col] = 1.0;

        // 前進代入
        std::vector<double> y_inv(p);
        for (std::size_t i = 0; i < p; ++i) {
            double sum = 0.0;
            for (std::size_t j = 0; j < i; ++j) {
                sum += L[i][j] * y_inv[j];
            }
            y_inv[i] = (e[i] - sum) / L[i][i];
        }

        // 後退代入
        for (std::size_t i = p; i > 0; --i) {
            std::size_t idx = i - 1;
            double sum = 0.0;
            for (std::size_t j = idx + 1; j < p; ++j) {
                sum += L[j][idx] * XtWX_inv[j][col];
            }
            XtWX_inv[idx][col] = (y_inv[idx] - sum) / L[idx][idx];
        }
    }

    return beta;
}

} // namespace detail

// ============================================================================
// IRLS Algorithm (反復再重み付け最小二乗法)
// ============================================================================

/**
 * @brief 一般化線形モデルのフィッティング
 *
 * IRLS（反復再重み付け最小二乗法）アルゴリズムを使用して
 * 一般化線形モデルのパラメータを推定します。
 *
 * @param X 説明変数の行列（切片は自動的に追加されます）
 * @param y 目的変数ベクトル
 * @param family 分布族（デフォルト: gaussian）
 * @param link リンク関数（デフォルト: identity）
 * @param max_iter 最大反復回数（デフォルト: 100）
 * @param tol 収束判定の許容誤差（デフォルト: 1e-8）
 * @return GLMの推定結果
 * @throws std::invalid_argument データが空の場合、XとYのサイズが一致しない場合、
 *         説明変数の数が不整合な場合、観測数が説明変数数以下の場合
 * @note IRLSアルゴリズムが max_iter 回以内に収束しなかった場合、
 *       converged は false に設定され、coefficient_se, z_statistics,
 *       p_values は NaN を含む場合があります。NaN は数値的問題により
 *       推定が不可能であることを示します（バグではありません）。
 *       これらのフィールドを使用する前に、必ず glm_result::converged を確認してください。
 */
inline glm_result glm_fit(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    distribution_family family = distribution_family::gaussian,
    link_function link = link_function::identity,
    std::size_t max_iter = 100,
    double tol = 1e-8)
{
    std::size_t n = X.size();
    if (n == 0) {
        throw std::invalid_argument("statcpp::glm_fit: empty data");
    }
    if (n != y.size()) {
        throw std::invalid_argument("statcpp::glm_fit: X and y must have same number of observations");
    }

    std::size_t p = X[0].size();
    for (const auto& row : X) {
        if (row.size() != p) {
            throw std::invalid_argument("statcpp::glm_fit: inconsistent number of predictors");
        }
    }

    std::size_t p_full = p + 1;  // 切片を含む
    if (n <= p_full) {
        throw std::invalid_argument("statcpp::glm_fit: need more observations than predictors");
    }

    // 設計行列（切片を追加）
    std::vector<std::vector<double>> X_design(n, std::vector<double>(p_full));
    for (std::size_t i = 0; i < n; ++i) {
        X_design[i][0] = 1.0;
        for (std::size_t j = 0; j < p; ++j) {
            X_design[i][j + 1] = X[i][j];
        }
    }

    // 初期値の設定
    std::vector<double> mu(n);
    std::vector<double> eta(n);

    // 応答変数の平均から初期値を設定
    double y_mean = statcpp::mean(y.begin(), y.end());
    double eta_init;

    switch (family) {
        case distribution_family::binomial:
            y_mean = std::max(0.01, std::min(0.99, y_mean));
            eta_init = detail::link_transform(y_mean, link);
            break;
        case distribution_family::poisson:
            y_mean = std::max(0.1, y_mean);
            eta_init = detail::link_transform(y_mean, link);
            break;
        default:
            eta_init = y_mean;
            break;
    }

    for (std::size_t i = 0; i < n; ++i) {
        eta[i] = eta_init;
        mu[i] = detail::inverse_link(eta[i], link);
    }

    // 係数の初期値
    std::vector<double> beta(p_full, 0.0);
    beta[0] = eta_init;

    std::vector<std::vector<double>> XtWX_inv;
    bool converged = false;
    std::size_t iter = 0;

    // IRLS反復
    for (iter = 0; iter < max_iter; ++iter) {
        // 重みと作業変数を計算
        std::vector<double> w(n);
        std::vector<double> z(n);

        for (std::size_t i = 0; i < n; ++i) {
            double var_i = detail::variance_function(mu[i], family);
            double g_prime = detail::link_derivative(mu[i], link);

            // 重み w_i = 1 / (V(mu_i) * g'(mu_i)^2)
            w[i] = 1.0 / (var_i * g_prime * g_prime);

            // 作業変数 z_i = eta_i + (y_i - mu_i) * g'(mu_i)
            z[i] = eta[i] + (y[i] - mu[i]) * g_prime;
        }

        // 重み付き最小二乗を解く
        std::vector<double> beta_new;
        try {
            beta_new = detail::solve_weighted_least_squares(X_design, z, w, XtWX_inv);
        } catch (const std::exception&) {
            break;  // 数値的に不安定な場合は終了
        }

        // 収束判定
        double max_change = 0.0;
        for (std::size_t j = 0; j < p_full; ++j) {
            double change = std::abs(beta_new[j] - beta[j]);
            if (std::abs(beta[j]) > 1.0) {
                change /= std::abs(beta[j]);
            }
            max_change = std::max(max_change, change);
        }

        beta = beta_new;

        // eta と mu を更新
        eta = detail::matrix_vector_multiply(X_design, beta);
        for (std::size_t i = 0; i < n; ++i) {
            mu[i] = detail::inverse_link(eta[i], link);
        }

        if (max_change < tol) {
            converged = true;
            ++iter;
            break;
        }
    }

    // 標準誤差の計算
    std::vector<double> coefficient_se(p_full, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> z_statistics(p_full, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> p_values(p_full, std::numeric_limits<double>::quiet_NaN());

    if (!XtWX_inv.empty()) {
        for (std::size_t j = 0; j < p_full; ++j) {
            coefficient_se[j] = std::sqrt(XtWX_inv[j][j]);
        }

        // z統計量とp値
        for (std::size_t j = 0; j < p_full; ++j) {
            z_statistics[j] = beta[j] / coefficient_se[j];
            p_values[j] = 2.0 * (1.0 - norm_cdf(std::abs(z_statistics[j])));
        }
    }

    // 逸脱度の計算
    double residual_deviance = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        residual_deviance += detail::deviance_residual(y[i], mu[i], family);
    }

    // ヌル逸脱度（切片のみのモデル）
    double null_deviance = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        null_deviance += detail::deviance_residual(y[i], y_mean, family);
    }

    // 対数尤度の計算
    double log_likelihood = 0.0;
    switch (family) {
        case distribution_family::gaussian:
            {
                double sigma2 = residual_deviance / static_cast<double>(n);
                log_likelihood = -0.5 * static_cast<double>(n) *
                    (std::log(2.0 * pi) + std::log(sigma2) + 1.0);
            }
            break;
        case distribution_family::binomial:
            for (std::size_t i = 0; i < n; ++i) {
                double mu_i = std::max(1e-10, std::min(1.0 - 1e-10, mu[i]));
                if (y[i] > 0.0) {
                    log_likelihood += y[i] * std::log(mu_i);
                }
                if (y[i] < 1.0) {
                    log_likelihood += (1.0 - y[i]) * std::log(1.0 - mu_i);
                }
            }
            break;
        case distribution_family::poisson:
            for (std::size_t i = 0; i < n; ++i) {
                log_likelihood += y[i] * std::log(std::max(1e-10, mu[i])) - mu[i]
                                  - std::lgamma(y[i] + 1.0);
            }
            break;
        default:
            log_likelihood = -0.5 * residual_deviance;
            break;
    }

    // AIC と BIC
    double n_d = static_cast<double>(n);
    double k = static_cast<double>(p_full);
    if (family == distribution_family::gaussian) {
        k += 1.0;  // sigma^2 も推定パラメータとしてカウント
    }
    double aic = -2.0 * log_likelihood + 2.0 * k;
    double bic = -2.0 * log_likelihood + k * std::log(n_d);

    return {
        beta, coefficient_se, z_statistics, p_values,
        null_deviance, residual_deviance,
        static_cast<double>(n - 1), static_cast<double>(n - p_full),
        aic, bic, log_likelihood,
        iter, converged,
        link, family
    };
}

// ============================================================================
// Logistic Regression (ロジスティック回帰)
// ============================================================================

/**
 * @brief ロジスティック回帰
 *
 * 二項分布とロジットリンク関数を使用した一般化線形モデルをフィットします。
 *
 * @param X 説明変数の行列（切片は自動的に追加されます）
 * @param y 目的変数ベクトル（0から1の範囲）
 * @param max_iter 最大反復回数（デフォルト: 100）
 * @param tol 収束判定の許容誤差（デフォルト: 1e-8）
 * @return GLMの推定結果
 * @throws std::invalid_argument yが[0,1]の範囲外の場合、Xに切片列が含まれている場合
 */
inline glm_result logistic_regression(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    std::size_t max_iter = 100,
    double tol = 1e-8)
{
    // X に切片列が含まれていないことを確認（linear_regression.hppから）
    statcpp::detail::validate_no_intercept_column(X, "logistic_regression");

    // y が 0 または 1 の範囲にあることを確認
    for (double yi : y) {
        if (yi < 0.0 || yi > 1.0) {
            throw std::invalid_argument("statcpp::logistic_regression: y must be in [0, 1]");
        }
    }

    return glm_fit(X, y, distribution_family::binomial, link_function::logit, max_iter, tol);
}

/**
 * @brief ロジスティック回帰での確率予測
 *
 * フィットされたロジスティック回帰モデルを使用して、
 * 新しいデータポイントに対する確率を予測します。
 *
 * @param model フィットされたGLMモデル（二項分布）
 * @param x 説明変数ベクトル
 * @return 予測確率（0から1の範囲）
 * @throws std::invalid_argument モデルが二項分布でない場合、xの次元が一致しない場合
 */
inline double predict_probability(const glm_result& model, const std::vector<double>& x)
{
    if (model.family != distribution_family::binomial) {
        throw std::invalid_argument("statcpp::predict_probability: model must be binomial");
    }
    if (x.size() + 1 != model.coefficients.size()) {
        throw std::invalid_argument("statcpp::predict_probability: x dimension mismatch");
    }

    double eta = model.coefficients[0];
    for (std::size_t i = 0; i < x.size(); ++i) {
        eta += model.coefficients[i + 1] * x[i];
    }

    return detail::inverse_link(eta, model.link);
}

/**
 * @brief オッズ比の計算
 *
 * ロジスティック回帰モデルの係数からオッズ比を計算します。
 *
 * @param model フィットされたロジスティック回帰モデル
 * @return 各説明変数のオッズ比ベクトル（切片を除く）
 * @throws std::invalid_argument モデルがロジスティック回帰でない場合
 */
inline std::vector<double> odds_ratios(const glm_result& model)
{
    if (model.family != distribution_family::binomial || model.link != link_function::logit) {
        throw std::invalid_argument("statcpp::odds_ratios: requires logistic regression model");
    }

    std::vector<double> or_values(model.coefficients.size() - 1);
    for (std::size_t i = 1; i < model.coefficients.size(); ++i) {
        or_values[i - 1] = std::exp(model.coefficients[i]);
    }
    return or_values;
}

/**
 * @brief オッズ比の信頼区間
 *
 * ロジスティック回帰モデルのオッズ比に対する信頼区間を計算します。
 *
 * @param model フィットされたロジスティック回帰モデル
 * @param confidence 信頼水準（デフォルト: 0.95）
 * @return 各説明変数のオッズ比の信頼区間（下限、上限）のペアベクトル
 * @throws std::invalid_argument モデルがロジスティック回帰でない場合、
 *         信頼水準が(0,1)の範囲外の場合
 */
inline std::vector<std::pair<double, double>> odds_ratios_ci(
    const glm_result& model, double confidence = 0.95)
{
    if (model.family != distribution_family::binomial || model.link != link_function::logit) {
        throw std::invalid_argument("statcpp::odds_ratios_ci: requires logistic regression model");
    }
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::odds_ratios_ci: confidence must be in (0, 1)");
    }

    double z = norm_quantile(1.0 - (1.0 - confidence) / 2.0);

    std::vector<std::pair<double, double>> ci(model.coefficients.size() - 1);
    for (std::size_t i = 1; i < model.coefficients.size(); ++i) {
        double beta = model.coefficients[i];
        double se = model.coefficient_se[i];
        double lower = std::exp(beta - z * se);
        double upper = std::exp(beta + z * se);
        ci[i - 1] = {lower, upper};
    }
    return ci;
}

// ============================================================================
// Poisson Regression (ポアソン回帰)
// ============================================================================

/**
 * @brief ポアソン回帰
 *
 * ポアソン分布と対数リンク関数を使用した一般化線形モデルをフィットします。
 * カウントデータの回帰分析に使用します。
 *
 * @param X 説明変数の行列（切片は自動的に追加されます）
 * @param y 目的変数ベクトル（非負のカウントデータ）
 * @param max_iter 最大反復回数（デフォルト: 100）
 * @param tol 収束判定の許容誤差（デフォルト: 1e-8）
 * @return GLMの推定結果
 * @throws std::invalid_argument yが負の場合、Xに切片列が含まれている場合
 */
inline glm_result poisson_regression(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    std::size_t max_iter = 100,
    double tol = 1e-8)
{
    // X に切片列が含まれていないことを確認
    statcpp::detail::validate_no_intercept_column(X, "poisson_regression");

    // y が非負であることを確認
    for (double yi : y) {
        if (yi < 0.0) {
            throw std::invalid_argument("statcpp::poisson_regression: y must be non-negative");
        }
    }

    return glm_fit(X, y, distribution_family::poisson, link_function::log, max_iter, tol);
}

/**
 * @brief ポアソン回帰での期待カウント予測
 *
 * フィットされたポアソン回帰モデルを使用して、
 * 新しいデータポイントに対する期待カウントを予測します。
 *
 * @param model フィットされたGLMモデル（ポアソン分布）
 * @param x 説明変数ベクトル
 * @return 予測された期待カウント
 * @throws std::invalid_argument モデルがポアソン分布でない場合、xの次元が一致しない場合
 */
inline double predict_count(const glm_result& model, const std::vector<double>& x)
{
    if (model.family != distribution_family::poisson) {
        throw std::invalid_argument("statcpp::predict_count: model must be Poisson");
    }
    if (x.size() + 1 != model.coefficients.size()) {
        throw std::invalid_argument("statcpp::predict_count: x dimension mismatch");
    }

    double eta = model.coefficients[0];
    for (std::size_t i = 0; i < x.size(); ++i) {
        eta += model.coefficients[i + 1] * x[i];
    }

    return detail::inverse_link(eta, model.link);
}

/**
 * @brief 発生率比（Incidence Rate Ratio）の計算
 *
 * ポアソン回帰モデルの係数から発生率比を計算します。
 *
 * @param model フィットされたポアソン回帰モデル
 * @return 各説明変数の発生率比ベクトル（切片を除く）
 * @throws std::invalid_argument モデルがポアソン回帰でない場合
 */
inline std::vector<double> incidence_rate_ratios(const glm_result& model)
{
    if (model.family != distribution_family::poisson || model.link != link_function::log) {
        throw std::invalid_argument("statcpp::incidence_rate_ratios: requires Poisson regression model");
    }

    std::vector<double> irr(model.coefficients.size() - 1);
    for (std::size_t i = 1; i < model.coefficients.size(); ++i) {
        irr[i - 1] = std::exp(model.coefficients[i]);
    }
    return irr;
}

// ============================================================================
// GLM Diagnostics
// ============================================================================

/**
 * @brief GLM残差構造体
 *
 * 一般化線形モデルの各種残差を格納します。
 */
struct glm_residuals {
    std::vector<double> response;       ///< 応答残差 (y - mu)
    std::vector<double> pearson;        ///< ピアソン残差
    std::vector<double> deviance;       ///< 逸脱度残差
    std::vector<double> working;        ///< 作業残差
};

/**
 * @brief GLM残差の計算
 *
 * フィットされたGLMモデルから各種残差を計算します。
 *
 * @param model フィットされたGLMモデル
 * @param X 説明変数の行列
 * @param y 目的変数ベクトル
 * @return 各種残差を含む構造体
 * @throws std::invalid_argument XとYのサイズが一致しない場合
 */
inline glm_residuals compute_glm_residuals(
    const glm_result& model,
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y)
{
    std::size_t n = X.size();
    if (n != y.size()) {
        throw std::invalid_argument("statcpp::compute_glm_residuals: X and y must have same length");
    }

    std::vector<double> response(n);
    std::vector<double> pearson(n);
    std::vector<double> deviance_res(n);
    std::vector<double> working(n);

    for (std::size_t i = 0; i < n; ++i) {
        // 線形予測子を計算
        double eta = model.coefficients[0];
        for (std::size_t j = 0; j < X[i].size(); ++j) {
            eta += model.coefficients[j + 1] * X[i][j];
        }

        double mu = detail::inverse_link(eta, model.link);

        // 応答残差
        response[i] = y[i] - mu;

        // ピアソン残差
        double var = detail::variance_function(mu, model.family);
        pearson[i] = response[i] / std::sqrt(var);

        // 逸脱度残差
        double d = detail::deviance_residual(y[i], mu, model.family);
        int sign = (y[i] >= mu) ? 1 : -1;
        deviance_res[i] = sign * std::sqrt(d);

        // 作業残差
        double g_prime = detail::link_derivative(mu, model.link);
        working[i] = response[i] * g_prime;
    }

    return {response, pearson, deviance_res, working};
}

/**
 * @brief 過分散の検定（ポアソン回帰用）
 *
 * ポアソン回帰モデルにおける過分散パラメータを計算します。
 * 値が1より大きい場合、過分散の存在を示唆します。
 *
 * @param model フィットされたポアソン回帰モデル
 * @param X 説明変数の行列
 * @param y 目的変数ベクトル
 * @return 過分散パラメータ（ピアソンカイ二乗統計量 / 残差自由度）
 * @throws std::invalid_argument モデルがポアソン分布でない場合
 */
inline double overdispersion_test(const glm_result& model,
                                   const std::vector<std::vector<double>>& X,
                                   const std::vector<double>& y)
{
    if (model.family != distribution_family::poisson) {
        throw std::invalid_argument("statcpp::overdispersion_test: requires Poisson model");
    }

    auto residuals = compute_glm_residuals(model, X, y);

    // ピアソンカイ二乗統計量
    double pearson_chi2 = 0.0;
    for (double r : residuals.pearson) {
        pearson_chi2 += r * r;
    }

    // 分散推定（過分散パラメータ）
    double dispersion = pearson_chi2 / model.df_residual;

    return dispersion;
}

/**
 * @brief 疑似決定係数（McFadden）
 *
 * McFaddenの疑似決定係数を計算します。
 * 1 - (残差逸脱度 / ヌル逸脱度) として定義されます。
 *
 * @param model フィットされたGLMモデル
 * @return McFaddenの疑似決定係数
 */
inline double pseudo_r_squared_mcfadden(const glm_result& model)
{
    return 1.0 - (model.residual_deviance / model.null_deviance);
}

/**
 * @brief 疑似決定係数（Nagelkerke）
 *
 * Nagelkerkeの疑似決定係数を計算します。
 * Cox-Snellの疑似決定係数を最大値1になるように調整したものです。
 *
 * 逸脱度と対数尤度の関係: deviance = -2 * (LL_model - LL_saturated) を用いて
 * LL_null = LL_saturated - null_deviance / 2 として帰無モデルの対数尤度を計算します。
 *
 * @param model フィットされたGLMモデル
 * @param y 目的変数ベクトル（非ガウス族の飽和モデル対数尤度計算に必要）
 * @param n サンプルサイズ
 * @return Nagelkerkeの疑似決定係数
 */
inline double pseudo_r_squared_nagelkerke(const glm_result& model,
                                           const std::vector<double>& y,
                                           std::size_t n)
{
    double n_d = static_cast<double>(n);
    double ll_model = model.log_likelihood;

    // 飽和モデルの対数尤度を計算
    double ll_saturated = 0.0;
    switch (model.family) {
        case distribution_family::gaussian:
            break;
        case distribution_family::binomial:
            // 0/1データの飽和モデル対数尤度は 0
            ll_saturated = 0.0;
            break;
        case distribution_family::poisson:
            for (std::size_t i = 0; i < n; ++i) {
                if (y[i] > 0.0) {
                    ll_saturated += y[i] * std::log(y[i]) - y[i] - std::lgamma(y[i] + 1.0);
                }
            }
            break;
        default:
            break;
    }

    double ll_null;
    if (model.family == distribution_family::gaussian) {
        ll_null = -model.null_deviance / 2.0;
    } else {
        ll_null = ll_saturated - model.null_deviance / 2.0;
    }

    double r2_cox_snell = 1.0 - std::exp(2.0 * (ll_null - ll_model) / n_d);
    double r2_max = 1.0 - std::exp(2.0 * ll_null / n_d);

    if (r2_max == 0.0) return 0.0;

    return r2_cox_snell / r2_max;
}

} // namespace statcpp
