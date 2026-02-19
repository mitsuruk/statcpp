/**
 * @file random_engine.hpp
 * @brief Random engine wrapper and utilities
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
 * @brief Default random engine type (Mersenne Twister 64-bit version)
 */
using default_random_engine = std::mt19937_64;

/**
 * @brief Singleton accessor for global random engine
 *
 * Returns a thread-local random engine.
 * Each thread can generate independent random sequences, making it thread-safe.
 *
 * @return Reference to thread-local random engine
 */
inline default_random_engine& get_random_engine()
{
    thread_local default_random_engine engine(std::random_device{}());
    return engine;
}

/**
 * @brief Set the seed of the random engine
 *
 * Sets the seed of the global random engine to the specified value.
 * Use this when reproducible random sequences are needed.
 *
 * @param seed Seed value to set
 */
inline void set_seed(std::uint64_t seed)
{
    get_random_engine().seed(seed);
}

/**
 * @brief Randomly reset the random engine seed
 *
 * Re-initializes the global random engine seed with a value obtained from a random device.
 * Use this when unpredictable random sequences are needed.
 */
inline void randomize_seed()
{
    get_random_engine().seed(std::random_device{}());
}

/**
 * @brief Type trait to determine if a type is a random engine
 *
 * A concept-like type trait for templates that accept specific engines.
 *
 * @tparam T Type to check
 */
template <typename T>
struct is_random_engine : std::false_type {};

/**
 * @brief Specialization for std::mt19937
 */
template <>
struct is_random_engine<std::mt19937> : std::true_type {};

/**
 * @brief Specialization for std::mt19937_64
 */
template <>
struct is_random_engine<std::mt19937_64> : std::true_type {};

/**
 * @brief Specialization for std::minstd_rand
 */
template <>
struct is_random_engine<std::minstd_rand> : std::true_type {};

/**
 * @brief Specialization for std::minstd_rand0
 */
template <>
struct is_random_engine<std::minstd_rand0> : std::true_type {};

/**
 * @brief Specialization for std::ranlux24_base
 */
template <>
struct is_random_engine<std::ranlux24_base> : std::true_type {};

/**
 * @brief Specialization for std::ranlux48_base
 */
template <>
struct is_random_engine<std::ranlux48_base> : std::true_type {};

/**
 * @brief Specialization for std::ranlux24
 */
template <>
struct is_random_engine<std::ranlux24> : std::true_type {};

/**
 * @brief Specialization for std::ranlux48
 */
template <>
struct is_random_engine<std::ranlux48> : std::true_type {};

/**
 * @brief Specialization for std::knuth_b
 */
template <>
struct is_random_engine<std::knuth_b> : std::true_type {};

/**
 * @brief Variable template version of is_random_engine
 *
 * @tparam T Type to check
 */
template <typename T>
inline constexpr bool is_random_engine_v = is_random_engine<T>::value;

} // namespace statcpp
