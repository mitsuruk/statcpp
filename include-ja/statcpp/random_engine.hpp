/**
 * @file random_engine.hpp
 * @brief 乱数エンジンのラッパーとユーティリティ
 */

#pragma once

#include <cstdint>
#include <random>
#include <type_traits>

namespace statcpp {

// ============================================================================
// Random Engine Wrapper
// ============================================================================

/**
 * @brief デフォルトの乱数エンジン型（Mersenne Twister 64ビット版）
 */
using default_random_engine = std::mt19937_64;

/**
 * @brief グローバル乱数エンジンのシングルトンアクセサ
 *
 * スレッドローカルな乱数エンジンを返します。
 * 各スレッドで独立した乱数シーケンスを生成でき、スレッドセーフです。
 *
 * @return スレッドローカルな乱数エンジンへの参照
 */
inline default_random_engine& get_random_engine()
{
    thread_local default_random_engine engine(std::random_device{}());
    return engine;
}

/**
 * @brief 乱数エンジンのシードを設定
 *
 * グローバル乱数エンジンのシードを指定された値に設定します。
 * 再現可能な乱数シーケンスが必要な場合に使用します。
 *
 * @param seed 設定するシード値
 */
inline void set_seed(std::uint64_t seed)
{
    get_random_engine().seed(seed);
}

/**
 * @brief 乱数エンジンのシードをランダムに再設定
 *
 * グローバル乱数エンジンのシードをランダムデバイスから取得した値で再初期化します。
 * 予測不可能な乱数シーケンスが必要な場合に使用します。
 */
inline void randomize_seed()
{
    get_random_engine().seed(std::random_device{}());
}

/**
 * @brief 乱数エンジンかどうかを判定する型トレイト
 *
 * 特定のエンジンを受け取るテンプレートのためのコンセプト的な型トレイトです。
 *
 * @tparam T 判定する型
 */
template <typename T>
struct is_random_engine : std::false_type {};

/**
 * @brief std::mt19937の特殊化
 */
template <>
struct is_random_engine<std::mt19937> : std::true_type {};

/**
 * @brief std::mt19937_64の特殊化
 */
template <>
struct is_random_engine<std::mt19937_64> : std::true_type {};

/**
 * @brief std::minstd_randの特殊化
 */
template <>
struct is_random_engine<std::minstd_rand> : std::true_type {};

/**
 * @brief std::minstd_rand0の特殊化
 */
template <>
struct is_random_engine<std::minstd_rand0> : std::true_type {};

/**
 * @brief std::ranlux24_baseの特殊化
 */
template <>
struct is_random_engine<std::ranlux24_base> : std::true_type {};

/**
 * @brief std::ranlux48_baseの特殊化
 */
template <>
struct is_random_engine<std::ranlux48_base> : std::true_type {};

/**
 * @brief std::ranlux24の特殊化
 */
template <>
struct is_random_engine<std::ranlux24> : std::true_type {};

/**
 * @brief std::ranlux48の特殊化
 */
template <>
struct is_random_engine<std::ranlux48> : std::true_type {};

/**
 * @brief std::knuth_bの特殊化
 */
template <>
struct is_random_engine<std::knuth_b> : std::true_type {};

/**
 * @brief is_random_engineの変数テンプレート版
 *
 * @tparam T 判定する型
 */
template <typename T>
inline constexpr bool is_random_engine_v = is_random_engine<T>::value;

} // namespace statcpp
