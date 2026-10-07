/**
 * @file nan_utils.hpp
 * @brief NaN(欠損値)処理の補助関数 (NaN (missing value) handling helpers)
 *
 * NaN 取り扱い方針(docs-ja/NAN_POLICY.md)を実装するため、公開関数が共通で使う内部補助関数です。
 * NaN の検出、除去、末尾への並べ替え、統一した例外メッセージでの拒否を提供します。
 *
 * このファイルの内容はすべて statcpp::detail にあり、公開 API には含まれません。
 */

#pragma once

#include <cmath>
#include <functional>
#include <cstddef>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace statcpp {
namespace detail {

// ============================================================================
// 検出 (Detection)
// ============================================================================

/**
 * @brief 値が NaN かどうかを判定する
 *
 * 浮動小数点型は std::isnan で判定します。それ以外の型は NaN を持てないため常に false を返します。
 * これにより、任意の値型を取るテンプレート関数(value_counts<T> など)でも同じ判定を使えます。
 *
 * @tparam T 値の型
 * @param value 判定する値
 * @return 浮動小数点の NaN なら true
 */
template <typename T>
bool is_nan_value(const T& value)
{
    if constexpr (std::is_floating_point_v<T>) {
        return std::isnan(value);
    } else {
        (void)value;
        return false;
    }
}

/**
 * @brief 範囲に NaN が含まれるかを判定する
 *
 * @tparam Iterator イテレータ型
 * @param first 範囲の先頭
 * @param last 範囲の末尾
 * @return いずれかの要素が NaN なら true
 */
template <typename Iterator>
bool has_nan(Iterator first, Iterator last)
{
    for (; first != last; ++first) {
        if (is_nan_value(*first)) {
            return true;
        }
    }
    return false;
}

/**
 * @brief 射影後の値に NaN が含まれるかを判定する
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数の型
 * @param first 範囲の先頭
 * @param last 範囲の末尾
 * @param proj 各要素に適用する射影
 * @return 射影後のいずれかの値が NaN なら true
 */
template <typename Iterator, typename Projection>
bool has_nan(Iterator first, Iterator last, Projection proj)
{
    for (; first != last; ++first) {
        if (is_nan_value(std::invoke(proj, *first))) {
            return true;
        }
    }
    return false;
}

// ============================================================================
// 除去 (Removal)
// ============================================================================

/**
 * @brief NaN を除いて範囲をコピーする
 *
 * 残る要素の相対的な順序は保たれます。R の t.test などの「NA を除去してから計算する」挙動を実現します。
 *
 * @tparam Iterator イテレータ型
 * @param first 範囲の先頭
 * @param last 範囲の末尾
 * @return NaN でない要素(元の順序のまま)
 */
template <typename Iterator>
auto drop_nan(Iterator first, Iterator last)
{
    using value_type = std::remove_cv_t<typename std::iterator_traits<Iterator>::value_type>;
    std::vector<value_type> result;
    for (; first != last; ++first) {
        if (!is_nan_value(*first)) {
            result.push_back(*first);
        }
    }
    return result;
}

/**
 * @brief 範囲を射影し、NaN でない射影後の値をコピーする
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数の型
 * @param first 範囲の先頭
 * @param last 範囲の末尾
 * @param proj 各要素に適用する射影
 * @return NaN でない射影後の値(元の順序のまま)
 */
template <typename Iterator, typename Projection>
auto drop_nan(Iterator first, Iterator last, Projection proj)
{
    using projected_type = std::decay_t<
        std::invoke_result_t<Projection&, typename std::iterator_traits<Iterator>::reference>>;
    std::vector<projected_type> result;
    for (; first != last; ++first) {
        auto value = std::invoke(proj, *first);
        if (!is_nan_value(value)) {
            result.push_back(value);
        }
    }
    return result;
}

/**
 * @brief 対応する 2 つの範囲をコピーし、NaN を含む対を除く
 *
 * どちらか一方の要素が NaN の対を除きます(R の complete.cases と同じ)。
 * 2 つ目の範囲は 1 つ目以上の要素数が必要で、先頭から std::distance(x_first, x_last) 個だけを読みます。
 *
 * @tparam IteratorX 1 つ目の範囲のイテレータ型
 * @tparam IteratorY 2 つ目の範囲のイテレータ型
 * @param x_first 1 つ目の範囲の先頭
 * @param x_last 1 つ目の範囲の末尾
 * @param y_first 2 つ目の範囲の先頭
 * @return NaN を含まない対(同じ長さの 2 つのベクトル)
 */
template <typename IteratorX, typename IteratorY>
auto drop_nan_pairs(IteratorX x_first, IteratorX x_last, IteratorY y_first)
{
    using x_type = std::remove_cv_t<typename std::iterator_traits<IteratorX>::value_type>;
    using y_type = std::remove_cv_t<typename std::iterator_traits<IteratorY>::value_type>;
    std::pair<std::vector<x_type>, std::vector<y_type>> result;
    for (; x_first != x_last; ++x_first, ++y_first) {
        if (!is_nan_value(*x_first) && !is_nan_value(*y_first)) {
            result.first.push_back(*x_first);
            result.second.push_back(*y_first);
        }
    }
    return result;
}

/**
 * @brief NaN 以外の値だけに要素ごとの変換を適用する
 *
 * NaN の要素を除いて残りに func を適用し、結果を元の位置に書き戻します。NaN の位置は NaN のままです。
 * R の p.adjust() の NA の扱い(補正は NA でない p 値だけを数える)と同じです。
 *
 * @tparam Function const std::vector<double>& を受け取り、同じ長さの std::vector<double> を返す呼び出し可能オブジェクト
 * @param values 入力値(NaN を含みうる)
 * @param func NaN 以外の値に適用する変換
 * @return NaN 以外の位置は変換結果、それ以外は NaN
 */
template <typename Function>
std::vector<double> map_non_nan(const std::vector<double>& values, Function func)
{
    std::vector<double> non_nan;
    std::vector<std::size_t> positions;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isnan(values[i])) {
            non_nan.push_back(values[i]);
            positions.push_back(i);
        }
    }
    std::vector<double> result(values.size(), std::numeric_limits<double>::quiet_NaN());
    if (non_nan.empty()) {
        return result;
    }
    const std::vector<double> mapped = func(non_nan);
    for (std::size_t k = 0; k < positions.size(); ++k) {
        result[positions[k]] = mapped[k];
    }
    return result;
}

// ============================================================================
// 行列 (行 = 観測)
// ============================================================================

/**
 * @brief 行列が NaN を含むか判定する
 *
 * @param matrix 行のベクトルとして表した行列
 * @return いずれかの要素が NaN なら true
 */
inline bool has_nan_matrix(const std::vector<std::vector<double>>& matrix)
{
    for (const auto& row : matrix) {
        if (has_nan(row.begin(), row.end())) {
            return true;
        }
    }
    return false;
}

/**
 * @brief 計画行列または応答変数が NaN を含むか判定する
 *
 * @param X 説明変数行列(行 = 観測)
 * @param y 応答変数ベクトル
 * @return X または y のいずれかの要素が NaN なら true
 */
inline bool has_nan_rows(const std::vector<std::vector<double>>& X, const std::vector<double>& y)
{
    return has_nan_matrix(X) || has_nan(y.begin(), y.end());
}

/**
 * @brief NaN を含まない観測(X の行と y の要素)をコピーする
 *
 * 応答変数または説明変数のいずれかが NaN の観測を除きます。R のモデル関数の
 * na.action = na.omit と同じです。X と y の行数は等しくなければなりません。
 *
 * @param X 説明変数行列(行 = 観測)
 * @param y 応答変数ベクトル
 * @return 欠損のない観測(行列と応答変数ベクトル)
 */
inline std::pair<std::vector<std::vector<double>>, std::vector<double>> drop_nan_rows(
    const std::vector<std::vector<double>>& X, const std::vector<double>& y)
{
    std::pair<std::vector<std::vector<double>>, std::vector<double>> result;
    for (std::size_t i = 0; i < X.size(); ++i) {
        if (!has_nan(X[i].begin(), X[i].end()) && !is_nan_value(y[i])) {
            result.first.push_back(X[i]);
            result.second.push_back(y[i]);
        }
    }
    return result;
}

/**
 * @brief NaN を含まない行をコピーする
 *
 * @param X 行列(行 = 観測)
 * @return 欠損のない行(元の順序のまま)
 */
inline std::vector<std::vector<double>> drop_nan_rows(const std::vector<std::vector<double>>& X)
{
    std::vector<std::vector<double>> result;
    for (const auto& row : X) {
        if (!has_nan(row.begin(), row.end())) {
            result.push_back(row);
        }
    }
    return result;
}

/**
 * @brief 行列が NaN を含めば例外を送出する
 *
 * @param matrix 行のベクトルとして表した行列
 * @param function_name 公開関数名(メッセージに使用)
 * @throws std::invalid_argument "statcpp::<function_name>: data contains NaN"
 */
inline void require_no_nan_matrix(const std::vector<std::vector<double>>& matrix, const char* function_name)
{
    if (has_nan_matrix(matrix)) {
        throw std::invalid_argument(std::string("statcpp::") + function_name + ": data contains NaN");
    }
}

// ============================================================================
// 並べ替え (Ordering)
// ============================================================================

/**
 * @brief NaN を末尾に置く昇順の比較
 *
 * NaN を含んでも strict weak ordering を満たします(operator< 単独では満たさず、std::sort が未定義動作になる)。
 * NaN でない値は通常どおり比較し、NaN はどの非 NaN 値よりも大きく、NaN 同士は同値として扱います。
 * R の order(na.last = TRUE) と同じく NaN を元の順序に保つには std::stable_sort を使います。
 */
struct nan_last_less {
    /**
     * @brief 2 つの値を比較する
     * @param a 左辺の値
     * @param b 右辺の値
     * @return a が b より前に並ぶなら true
     */
    template <typename T>
    bool operator()(const T& a, const T& b) const
    {
        const bool a_nan = is_nan_value(a);
        const bool b_nan = is_nan_value(b);
        if (a_nan || b_nan) {
            return !a_nan && b_nan;
        }
        return a < b;
    }
};

/**
 * @brief NaN を末尾に置く降順の比較
 *
 * nan_last_less の降順版です。R の order(decreasing = TRUE, na.last = TRUE) と同じく、NaN は末尾に置きます。
 */
struct nan_last_greater {
    /**
     * @brief 2 つの値を比較する
     * @param a 左辺の値
     * @param b 右辺の値
     * @return a が b より前に並ぶなら true
     */
    template <typename T>
    bool operator()(const T& a, const T& b) const
    {
        const bool a_nan = is_nan_value(a);
        const bool b_nan = is_nan_value(b);
        if (a_nan || b_nan) {
            return !a_nan && b_nan;
        }
        return a > b;
    }
};

// ============================================================================
// 拒否 (Rejection)
// ============================================================================

/**
 * @brief データ範囲に NaN が含まれれば例外を送出する
 *
 * 欠損した結果を表せない関数(整数・カテゴリを返す関数)と、R の対応関数がエラーになる関数で使います。
 *
 * @tparam Iterator イテレータ型
 * @param first 範囲の先頭
 * @param last 範囲の末尾
 * @param function_name 公開関数名(メッセージに使用)
 * @throws std::invalid_argument "statcpp::<function_name>: data contains NaN"
 */
template <typename Iterator>
void require_no_nan(Iterator first, Iterator last, const char* function_name)
{
    if (has_nan(first, last)) {
        throw std::invalid_argument(std::string("statcpp::") + function_name + ": data contains NaN");
    }
}

/**
 * @brief パラメータが NaN なら例外を送出する
 *
 * NaN との比較はすべて偽になるため、NaN のパラメータは `sigma <= 0` のような範囲検査をすり抜けます。
 * それらの検査より前に呼び出してください。
 *
 * @param value パラメータの値
 * @param function_name 公開関数名(メッセージに使用)
 * @param parameter_name パラメータ名(メッセージに使用)
 * @throws std::invalid_argument "statcpp::<function_name>: <parameter_name> is NaN"
 */
inline void require_param_not_nan(double value, const char* function_name, const char* parameter_name)
{
    if (std::isnan(value)) {
        throw std::invalid_argument(std::string("statcpp::") + function_name + ": " + parameter_name +
                                    " is NaN");
    }
}

} // namespace detail
} // namespace statcpp
