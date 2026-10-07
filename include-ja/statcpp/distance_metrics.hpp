/**
 * @file distance_metrics.hpp
 * @brief 距離指標と類似度指標
 *
 * ユークリッド距離,マンハッタン距離,コサイン類似度など,
 * 2つの系列間の距離・類似度を計算する関数を提供します。
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <iterator>
#include <vector>

#include "statcpp/nan_utils.hpp"

namespace statcpp {

/**
 * @brief ユークリッド距離 (L2ノルム)
 *
 * n次元空間の2点間のユークリッド距離を計算します:
 * d(x, y) = sqrt(sum((x_i - y_i)^2))
 *
 * @tparam Iterator1 第1系列のランダムアクセスイテレータ型
 * @tparam Iterator2 第2系列のランダムアクセスイテレータ型
 * @param first1 第1系列の開始イテレータ
 * @param last1 第1系列の終了イテレータ
 * @param first2 第2系列の開始イテレータ
 * @param last2 第2系列の終了イテレータ
 * @return ユークリッド距離
 * @throw std::invalid_argument 系列の長さが異なる場合
 *
 * @note R の dist() と同じく、差が NaN の座標を除き、和を n / n_used 倍に補正する。使える座標がなければ NaN を返す(docs-ja/NAN_POLICY.md)。
 */
template <typename Iterator1, typename Iterator2>
double euclidean_distance(Iterator1 first1, Iterator1 last1,
                          Iterator2 first2, Iterator2 last2)
{
    auto n1 = std::distance(first1, last1);
    auto n2 = std::distance(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::euclidean_distance: sequences must have the same length");
    }
    if (n1 == 0) {
        throw std::invalid_argument("statcpp::euclidean_distance: empty sequences");
    }

    // R の dist() と同じく、差が NaN の座標は除き、和を n / n_used 倍に補正する。
    // 使える座標がなければ NaN を返す(docs-ja/NAN_POLICY.md)
    double sum_sq = 0.0;
    std::size_t n_used = 0;
    auto it1 = first1;
    auto it2 = first2;

    while (it1 != last1) {
        double diff = static_cast<double>(*it1) - static_cast<double>(*it2);
        if (!std::isnan(diff)) {
            sum_sq += diff * diff;
            ++n_used;
        }
        ++it1;
        ++it2;
    }
    if (n_used == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (n_used < static_cast<std::size_t>(n1)) {
        sum_sq *= static_cast<double>(n1) / static_cast<double>(n_used);
    }

    return std::sqrt(sum_sq);
}

/**
 * @brief ユークリッド距離 (射影付き)
 *
 * @note R の dist() と同じく、差が NaN の座標を除き、和を n / n_used 倍に補正する。使える座標がなければ NaN を返す(docs-ja/NAN_POLICY.md)。
 */
template <typename Iterator1, typename Iterator2, typename Proj1, typename Proj2>
double euclidean_distance(Iterator1 first1, Iterator1 last1,
                          Iterator2 first2, Iterator2 last2,
                          Proj1 proj1, Proj2 proj2)
{
    auto n1 = std::distance(first1, last1);
    auto n2 = std::distance(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::euclidean_distance: sequences must have the same length");
    }
    if (n1 == 0) {
        throw std::invalid_argument("statcpp::euclidean_distance: empty sequences");
    }

    // R の dist() と同じく、差が NaN の座標は除き、和を n / n_used 倍に補正する。
    // 使える座標がなければ NaN を返す(docs-ja/NAN_POLICY.md)
    double sum_sq = 0.0;
    std::size_t n_used = 0;
    auto it1 = first1;
    auto it2 = first2;

    while (it1 != last1) {
        double v1 = static_cast<double>(proj1(*it1));
        double v2 = static_cast<double>(proj2(*it2));
        double diff = v1 - v2;
        if (!std::isnan(diff)) {
            sum_sq += diff * diff;
            ++n_used;
        }
        ++it1;
        ++it2;
    }
    if (n_used == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (n_used < static_cast<std::size_t>(n1)) {
        sum_sq *= static_cast<double>(n1) / static_cast<double>(n_used);
    }

    return std::sqrt(sum_sq);
}

/**
 * @brief マンハッタン距離 (L1ノルム, タクシー距離)
 *
 * 2点間のマンハッタン距離を計算します:
 * d(x, y) = sum(|x_i - y_i|)
 *
 * @tparam Iterator1 第1系列のランダムアクセスイテレータ型
 * @tparam Iterator2 第2系列のランダムアクセスイテレータ型
 * @param first1 第1系列の開始イテレータ
 * @param last1 第1系列の終了イテレータ
 * @param first2 第2系列の開始イテレータ
 * @param last2 第2系列の終了イテレータ
 * @return マンハッタン距離
 * @throw std::invalid_argument 系列の長さが異なる場合
 *
 * @note R の dist() と同じく、差が NaN の座標を除き、和を n / n_used 倍に補正する。使える座標がなければ NaN を返す(docs-ja/NAN_POLICY.md)。
 */
template <typename Iterator1, typename Iterator2>
double manhattan_distance(Iterator1 first1, Iterator1 last1,
                          Iterator2 first2, Iterator2 last2)
{
    auto n1 = std::distance(first1, last1);
    auto n2 = std::distance(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::manhattan_distance: sequences must have the same length");
    }
    if (n1 == 0) {
        throw std::invalid_argument("statcpp::manhattan_distance: empty sequences");
    }

    // R の dist() と同じく、差が NaN の座標は除き、和を n / n_used 倍に補正する。
    // 使える座標がなければ NaN を返す(docs-ja/NAN_POLICY.md)
    double sum_abs = 0.0;
    std::size_t n_used = 0;
    auto it1 = first1;
    auto it2 = first2;

    while (it1 != last1) {
        double diff = static_cast<double>(*it1) - static_cast<double>(*it2);
        if (!std::isnan(diff)) {
            sum_abs += std::abs(diff);
            ++n_used;
        }
        ++it1;
        ++it2;
    }
    if (n_used == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (n_used < static_cast<std::size_t>(n1)) {
        sum_abs *= static_cast<double>(n1) / static_cast<double>(n_used);
    }

    return sum_abs;
}

/**
 * @brief マンハッタン距離 (射影付き)
 *
 * @note R の dist() と同じく、差が NaN の座標を除き、和を n / n_used 倍に補正する。使える座標がなければ NaN を返す(docs-ja/NAN_POLICY.md)。
 */
template <typename Iterator1, typename Iterator2, typename Proj1, typename Proj2>
double manhattan_distance(Iterator1 first1, Iterator1 last1,
                          Iterator2 first2, Iterator2 last2,
                          Proj1 proj1, Proj2 proj2)
{
    auto n1 = std::distance(first1, last1);
    auto n2 = std::distance(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::manhattan_distance: sequences must have the same length");
    }
    if (n1 == 0) {
        throw std::invalid_argument("statcpp::manhattan_distance: empty sequences");
    }

    // R の dist() と同じく、差が NaN の座標は除き、和を n / n_used 倍に補正する。
    // 使える座標がなければ NaN を返す(docs-ja/NAN_POLICY.md)
    double sum_abs = 0.0;
    std::size_t n_used = 0;
    auto it1 = first1;
    auto it2 = first2;

    while (it1 != last1) {
        double v1 = static_cast<double>(proj1(*it1));
        double v2 = static_cast<double>(proj2(*it2));
        double diff = v1 - v2;
        if (!std::isnan(diff)) {
            sum_abs += std::abs(diff);
            ++n_used;
        }
        ++it1;
        ++it2;
    }
    if (n_used == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (n_used < static_cast<std::size_t>(n1)) {
        sum_abs *= static_cast<double>(n1) / static_cast<double>(n_used);
    }

    return sum_abs;
}

/**
 * @brief コサイン類似度
 *
 * 2つのベクトル間の角度のコサインを計算します:
 * similarity(x, y) = (x . y) / (||x|| * ||y||)
 *
 * 戻り値は [-1, 1] の範囲:
 * - 1: 同じ方向を向いている
 * - 0: 直交している
 * - -1: 反対方向を向いている
 *
 * @tparam Iterator1 第1系列のランダムアクセスイテレータ型
 * @tparam Iterator2 第2系列のランダムアクセスイテレータ型
 * @param first1 第1系列の開始イテレータ
 * @param last1 第1系列の終了イテレータ
 * @param first2 第2系列の開始イテレータ
 * @param last2 第2系列の終了イテレータ
 * @return コサイン類似度
 * @throw std::invalid_argument 系列の長さが異なる場合,またはゼロベクトルの場合
 *
 * @note NaN を含む組は除く(ペアワイズ除去)。使える組がなければ NaN を返す(docs-ja/NAN_POLICY.md)。
 */
template <typename Iterator1, typename Iterator2>
double cosine_similarity(Iterator1 first1, Iterator1 last1,
                         Iterator2 first2, Iterator2 last2)
{
    auto n1 = std::distance(first1, last1);
    auto n2 = std::distance(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::cosine_similarity: sequences must have the same length");
    }
    if (n1 == 0) {
        throw std::invalid_argument("statcpp::cosine_similarity: empty sequences");
    }

    double dot_product = 0.0;
    double norm1_sq = 0.0;
    double norm2_sq = 0.0;
    std::size_t n_used = 0;  // NaN を含まない組の数

    auto it1 = first1;
    auto it2 = first2;

    while (it1 != last1) {
        double v1 = static_cast<double>(*it1);
        double v2 = static_cast<double>(*it2);

        // NaN を含む組は除く(proxy::simil と同じペアワイズ除去、docs-ja/NAN_POLICY.md)
        if (!std::isnan(v1) && !std::isnan(v2)) {
            dot_product += v1 * v2;
            norm1_sq += v1 * v1;
            norm2_sq += v2 * v2;
            ++n_used;
        }

        ++it1;
        ++it2;
    }

    if (n_used == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (norm1_sq == 0.0 || norm2_sq == 0.0) {
        throw std::invalid_argument("statcpp::cosine_similarity: zero vector encountered");
    }

    return dot_product / (std::sqrt(norm1_sq) * std::sqrt(norm2_sq));
}

/**
 * @brief コサイン類似度 (射影付き)
 *
 * @note NaN を含む組は除く(ペアワイズ除去)。使える組がなければ NaN を返す(docs-ja/NAN_POLICY.md)。
 */
template <typename Iterator1, typename Iterator2, typename Proj1, typename Proj2>
double cosine_similarity(Iterator1 first1, Iterator1 last1,
                         Iterator2 first2, Iterator2 last2,
                         Proj1 proj1, Proj2 proj2)
{
    auto n1 = std::distance(first1, last1);
    auto n2 = std::distance(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::cosine_similarity: sequences must have the same length");
    }
    if (n1 == 0) {
        throw std::invalid_argument("statcpp::cosine_similarity: empty sequences");
    }

    double dot_product = 0.0;
    double norm1_sq = 0.0;
    double norm2_sq = 0.0;
    std::size_t n_used = 0;  // NaN を含まない組の数

    auto it1 = first1;
    auto it2 = first2;

    while (it1 != last1) {
        double v1 = static_cast<double>(proj1(*it1));
        double v2 = static_cast<double>(proj2(*it2));

        // NaN を含む組は除く(proxy::simil と同じペアワイズ除去、docs-ja/NAN_POLICY.md)
        if (!std::isnan(v1) && !std::isnan(v2)) {
            dot_product += v1 * v2;
            norm1_sq += v1 * v1;
            norm2_sq += v2 * v2;
            ++n_used;
        }

        ++it1;
        ++it2;
    }

    if (n_used == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (norm1_sq == 0.0 || norm2_sq == 0.0) {
        throw std::invalid_argument("statcpp::cosine_similarity: zero vector encountered");
    }

    return dot_product / (std::sqrt(norm1_sq) * std::sqrt(norm2_sq));
}

/**
 * @brief コサイン距離
 *
 * コサイン距離 = 1 - コサイン類似度 として定義されます.
 * 戻り値は [0, 2] の範囲で,0は同一方向を意味します.
 *
 * @tparam Iterator1 第1系列のランダムアクセスイテレータ型
 * @tparam Iterator2 第2系列のランダムアクセスイテレータ型
 * @param first1 第1系列の開始イテレータ
 * @param last1 第1系列の終了イテレータ
 * @param first2 第2系列の開始イテレータ
 * @param last2 第2系列の終了イテレータ
 * @return コサイン距離
 *
 * @note NaN を含む組は除く(ペアワイズ除去)。使える組がなければ NaN を返す(docs-ja/NAN_POLICY.md)。
 */
template <typename Iterator1, typename Iterator2>
double cosine_distance(Iterator1 first1, Iterator1 last1,
                       Iterator2 first2, Iterator2 last2)
{
    return 1.0 - cosine_similarity(first1, last1, first2, last2);
}

/**
 * @brief コサイン距離 (射影付き)
 *
 * @note NaN を含む組は除く(ペアワイズ除去)。使える組がなければ NaN を返す(docs-ja/NAN_POLICY.md)。
 */
template <typename Iterator1, typename Iterator2, typename Proj1, typename Proj2>
double cosine_distance(Iterator1 first1, Iterator1 last1,
                       Iterator2 first2, Iterator2 last2,
                       Proj1 proj1, Proj2 proj2)
{
    return 1.0 - cosine_similarity(first1, last1, first2, last2, proj1, proj2);
}

/**
 * @brief マハラノビス距離
 *
 * 点と分布の間のマハラノビス距離を計算します.
 * データの相関を考慮し,スケール不変です.
 *
 * 点 x と平均 mu,共分散行列 Sigma の分布に対して:
 * d(x, mu) = sqrt((x - mu)^T Sigma^{-1} (x - mu))
 *
 * 現在の実装は2次元データのみに対応しています.
 *
 * @param x 距離を測る点
 * @param mean 分布の平均
 * @param cov_matrix 共分散行列 (2x2)
 * @return マハラノビス距離
 * @throw std::invalid_argument 次元が一致しない場合,または共分散行列が特異な場合
 */
inline double mahalanobis_distance(const std::vector<double>& x,
                                   const std::vector<double>& mean,
                                   const std::vector<std::vector<double>>& cov_matrix)
{
    if (x.size() != mean.size()) {
        throw std::invalid_argument("statcpp::mahalanobis_distance: x and mean must have same dimension");
    }
    if (x.size() != 2) {
        throw std::invalid_argument("statcpp::mahalanobis_distance: only 2D is currently supported");
    }
    if (cov_matrix.size() != 2 || cov_matrix[0].size() != 2 || cov_matrix[1].size() != 2) {
        throw std::invalid_argument("statcpp::mahalanobis_distance: covariance matrix must be 2x2");
    }

    // 差分ベクトルを計算: d = x - mean
    double d0 = x[0] - mean[0];
    double d1 = x[1] - mean[1];

    // 2x2共分散行列の逆行列を計算
    // Sigma = [a b]
    //         [c d]
    // Sigma^{-1} = (1/det) * [d  -b]
    //                        [-c  a]

    double a = cov_matrix[0][0];
    double b = cov_matrix[0][1];
    double c = cov_matrix[1][0];
    double d = cov_matrix[1][1];

    double det = a * d - b * c;

    if (std::abs(det) < 1e-10) {
        throw std::invalid_argument("statcpp::mahalanobis_distance: covariance matrix is singular");
    }

    // 逆行列の要素
    double inv_a = d / det;
    double inv_b = -b / det;
    double inv_c = -c / det;
    double inv_d = a / det;

    // d^T Sigma^{-1} d を計算
    // まず Sigma^{-1} d を計算
    double temp0 = inv_a * d0 + inv_b * d1;
    double temp1 = inv_c * d0 + inv_d * d1;

    // 次に d^T (Sigma^{-1} d) を計算
    double mahalanobis_sq = d0 * temp0 + d1 * temp1;

    if (mahalanobis_sq < 0.0) {
        // 数値誤差により負になる場合を防止
        mahalanobis_sq = 0.0;
    }

    return std::sqrt(mahalanobis_sq);
}

/**
 * @brief ミンコフスキー距離 (一般化Lp距離)
 *
 * パラメータ p のミンコフスキー距離を計算します:
 * d(x, y) = (sum(|x_i - y_i|^p))^(1/p)
 *
 * 特殊ケース:
 * - p = 1: マンハッタン距離
 * - p = 2: ユークリッド距離
 * - p -> inf: チェビシェフ距離 (max |x_i - y_i|)
 *
 * @tparam Iterator1 第1系列のランダムアクセスイテレータ型
 * @tparam Iterator2 第2系列のランダムアクセスイテレータ型
 * @param first1 第1系列の開始イテレータ
 * @param last1 第1系列の終了イテレータ
 * @param first2 第2系列の開始イテレータ
 * @param last2 第2系列の終了イテレータ
 * @param p ミンコフスキー距離の次数 (1以上)
 * @return ミンコフスキー距離
 * @throws std::invalid_argument p が NaN の場合(docs-ja/NAN_POLICY.md 第 5 節)
 * @throw std::invalid_argument 系列の長さが異なる場合,またはp < 1の場合
 *
 * @note R の dist() と同じく、差が NaN の座標を除き、和を n / n_used 倍に補正する。使える座標がなければ NaN を返す(docs-ja/NAN_POLICY.md)。
 */
template <typename Iterator1, typename Iterator2>
double minkowski_distance(Iterator1 first1, Iterator1 last1,
                          Iterator2 first2, Iterator2 last2,
                          double p)
{
    detail::require_param_not_nan(p, "minkowski_distance", "p");
    if (p < 1.0) {
        throw std::invalid_argument("statcpp::minkowski_distance: p must be >= 1");
    }

    auto n1 = std::distance(first1, last1);
    auto n2 = std::distance(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::minkowski_distance: sequences must have the same length");
    }
    if (n1 == 0) {
        throw std::invalid_argument("statcpp::minkowski_distance: empty sequences");
    }

    // R の dist() と同じく、差が NaN の座標は除き、和を n / n_used 倍に補正する。
    // 使える座標がなければ NaN を返す(docs-ja/NAN_POLICY.md)
    double sum = 0.0;
    std::size_t n_used = 0;
    auto it1 = first1;
    auto it2 = first2;

    while (it1 != last1) {
        double diff = std::abs(static_cast<double>(*it1) - static_cast<double>(*it2));
        if (!std::isnan(diff)) {
            sum += std::pow(diff, p);
            ++n_used;
        }
        ++it1;
        ++it2;
    }
    if (n_used == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (n_used < static_cast<std::size_t>(n1)) {
        sum *= static_cast<double>(n1) / static_cast<double>(n_used);
    }

    return std::pow(sum, 1.0 / p);
}

/**
 * @brief ミンコフスキー距離 (射影付き)
 *
 * @throws std::invalid_argument p が NaN の場合(docs-ja/NAN_POLICY.md 第 5 節)
 *
 * @note R の dist() と同じく、差が NaN の座標を除き、和を n / n_used 倍に補正する。使える座標がなければ NaN を返す(docs-ja/NAN_POLICY.md)。
 */
template <typename Iterator1, typename Iterator2, typename Proj1, typename Proj2>
double minkowski_distance(Iterator1 first1, Iterator1 last1,
                          Iterator2 first2, Iterator2 last2,
                          double p, Proj1 proj1, Proj2 proj2)
{
    detail::require_param_not_nan(p, "minkowski_distance", "p");
    if (p < 1.0) {
        throw std::invalid_argument("statcpp::minkowski_distance: p must be >= 1");
    }

    auto n1 = std::distance(first1, last1);
    auto n2 = std::distance(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::minkowski_distance: sequences must have the same length");
    }
    if (n1 == 0) {
        throw std::invalid_argument("statcpp::minkowski_distance: empty sequences");
    }

    // R の dist() と同じく、差が NaN の座標は除き、和を n / n_used 倍に補正する。
    // 使える座標がなければ NaN を返す(docs-ja/NAN_POLICY.md)
    double sum = 0.0;
    std::size_t n_used = 0;
    auto it1 = first1;
    auto it2 = first2;

    while (it1 != last1) {
        double v1 = static_cast<double>(proj1(*it1));
        double v2 = static_cast<double>(proj2(*it2));
        double diff = std::abs(v1 - v2);
        if (!std::isnan(diff)) {
            sum += std::pow(diff, p);
            ++n_used;
        }
        ++it1;
        ++it2;
    }
    if (n_used == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (n_used < static_cast<std::size_t>(n1)) {
        sum *= static_cast<double>(n1) / static_cast<double>(n_used);
    }

    return std::pow(sum, 1.0 / p);
}

/**
 * @brief チェビシェフ距離 (L-inf ノルム, 最大距離)
 *
 * 絶対差の最大値を計算します:
 * d(x, y) = max(|x_i - y_i|)
 *
 * @tparam Iterator1 第1系列のランダムアクセスイテレータ型
 * @tparam Iterator2 第2系列のランダムアクセスイテレータ型
 * @param first1 第1系列の開始イテレータ
 * @param last1 第1系列の終了イテレータ
 * @param first2 第2系列の開始イテレータ
 * @param last2 第2系列の終了イテレータ
 * @return チェビシェフ距離
 * @throw std::invalid_argument 系列の長さが異なる場合
 */
template <typename Iterator1, typename Iterator2>
double chebyshev_distance(Iterator1 first1, Iterator1 last1,
                          Iterator2 first2, Iterator2 last2)
{
    auto n1 = std::distance(first1, last1);
    auto n2 = std::distance(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::chebyshev_distance: sequences must have the same length");
    }
    if (n1 == 0) {
        throw std::invalid_argument("statcpp::chebyshev_distance: empty sequences");
    }

    double max_diff = 0.0;
    auto it1 = first1;
    auto it2 = first2;

    while (it1 != last1) {
        double diff = std::abs(static_cast<double>(*it1) - static_cast<double>(*it2));
        if (diff > max_diff) {
            max_diff = diff;
        }
        ++it1;
        ++it2;
    }

    return max_diff;
}

/**
 * @brief チェビシェフ距離 (射影付き)
 */
template <typename Iterator1, typename Iterator2, typename Proj1, typename Proj2>
double chebyshev_distance(Iterator1 first1, Iterator1 last1,
                          Iterator2 first2, Iterator2 last2,
                          Proj1 proj1, Proj2 proj2)
{
    auto n1 = std::distance(first1, last1);
    auto n2 = std::distance(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::chebyshev_distance: sequences must have the same length");
    }
    if (n1 == 0) {
        throw std::invalid_argument("statcpp::chebyshev_distance: empty sequences");
    }

    double max_diff = 0.0;
    auto it1 = first1;
    auto it2 = first2;

    while (it1 != last1) {
        double v1 = static_cast<double>(proj1(*it1));
        double v2 = static_cast<double>(proj2(*it2));
        double diff = std::abs(v1 - v2);
        if (diff > max_diff) {
            max_diff = diff;
        }
        ++it1;
        ++it2;
    }

    return max_diff;
}

} // namespace statcpp
