/**
 * @file survival.hpp
 * @brief 生存時間解析の関数
 *
 * Kaplan-Meier推定、Log-rank検定、Nelson-Aalen推定など、
 * 生存時間解析に関する関数を提供します。
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "statcpp/continuous_distributions.hpp"

namespace statcpp {

// ============================================================================
// Kaplan-Meier Estimator
// ============================================================================

/**
 * @brief Kaplan-Meier 推定の結果
 *
 * 生存曲線の推定結果を保持します。
 */
struct kaplan_meier_result {
    std::vector<double> times;              ///< イベント時刻
    std::vector<double> survival;           ///< 生存確率
    std::vector<double> se;                 ///< 標準誤差（Greenwood の公式）
    std::vector<double> ci_lower;           ///< 95%信頼区間下限
    std::vector<double> ci_upper;           ///< 95%信頼区間上限
    std::vector<std::size_t> n_at_risk;     ///< リスク集合のサイズ
    std::vector<std::size_t> n_events;      ///< 各時点でのイベント数
    std::vector<std::size_t> n_censored;    ///< 各時点での打ち切り数
};

/**
 * @brief Kaplan-Meier 生存曲線を推定
 *
 * 打ち切りデータを含む生存時間データから生存曲線を推定します。
 * Greenwood の公式により標準誤差を計算し、95%信頼区間を提供します。
 *
 * @param times 観測時間のベクトル
 * @param events イベント発生フラグ（true = イベント発生、false = 打ち切り）
 * @return Kaplan-Meier推定の結果
 * @throws std::invalid_argument timesとeventsのサイズが異なる場合、またはデータが空の場合
 */
inline kaplan_meier_result kaplan_meier(
    const std::vector<double>& times,
    const std::vector<bool>& events)
{
    if (times.size() != events.size()) {
        throw std::invalid_argument("statcpp::kaplan_meier: times and events must have same length");
    }
    if (times.empty()) {
        throw std::invalid_argument("statcpp::kaplan_meier: empty data");
    }

    std::size_t n = times.size();

    // インデックスを時間でソート
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&times](std::size_t i, std::size_t j) {
                  return times[i] < times[j];
              });

    kaplan_meier_result result;
    result.times.push_back(0.0);
    result.survival.push_back(1.0);
    result.se.push_back(0.0);
    result.n_at_risk.push_back(n);
    result.n_events.push_back(0);
    result.n_censored.push_back(0);

    double S = 1.0;  // 累積生存確率
    double var_sum = 0.0;  // Greenwood の分散項の累積和
    std::size_t n_risk = n;  // リスク集合のサイズ

    std::size_t i = 0;
    while (i < n) {
        double t = times[indices[i]];

        // この時点でのイベント数と打ち切り数をカウント
        std::size_t d = 0;  // イベント数
        std::size_t c = 0;  // 打ち切り数

        while (i < n && times[indices[i]] == t) {
            if (events[indices[i]]) {
                d++;
            } else {
                c++;
            }
            i++;
        }

        // イベントがあった場合のみ生存確率を更新
        if (d > 0) {
            double q = static_cast<double>(d) / static_cast<double>(n_risk);
            S *= (1.0 - q);

            // Greenwood の分散
            if (n_risk > d) {
                var_sum += static_cast<double>(d) /
                          (static_cast<double>(n_risk) * static_cast<double>(n_risk - d));
            }

            double se = S * std::sqrt(var_sum);

            // 95% 信頼区間（対数変換）
            double z = 1.96;
            double ci_lower, ci_upper;
            if (S > 0 && S < 1) {
                double log_S = std::log(S);
                double se_log = se / S;
                ci_lower = std::exp(log_S - z * se_log);
                ci_upper = std::exp(log_S + z * se_log);
                ci_lower = std::max(0.0, ci_lower);
                ci_upper = std::min(1.0, ci_upper);
            } else {
                ci_lower = S;
                ci_upper = S;
            }

            result.times.push_back(t);
            result.survival.push_back(S);
            result.se.push_back(se);
            result.ci_lower.push_back(ci_lower);
            result.ci_upper.push_back(ci_upper);
            result.n_at_risk.push_back(n_risk);
            result.n_events.push_back(d);
            result.n_censored.push_back(c);
        }

        n_risk -= (d + c);
    }

    // 最初の点の信頼区間を設定
    result.ci_lower.insert(result.ci_lower.begin(), 1.0);
    result.ci_upper.insert(result.ci_upper.begin(), 1.0);

    return result;
}

// ============================================================================
// Log-rank Test
// ============================================================================

/**
 * @brief Log-rank 検定の結果
 *
 * 2群の生存曲線の比較検定の結果を保持します。
 */
struct logrank_result {
    double statistic;     ///< 検定統計量（χ²）
    double p_value;       ///< p値
    std::size_t df;       ///< 自由度
    double expected1;     ///< グループ1の期待イベント数
    double expected2;     ///< グループ2の期待イベント数
    std::size_t observed1; ///< グループ1の観測イベント数
    std::size_t observed2; ///< グループ2の観測イベント数
};

/**
 * @brief Log-rank 検定（2群の生存曲線の比較）
 *
 * 2つのグループの生存曲線が等しいかどうかを検定します。
 * 各時点でのリスク集合を考慮した非パラメトリック検定です。
 *
 * @param times1 グループ1の観測時間
 * @param events1 グループ1のイベント発生フラグ
 * @param times2 グループ2の観測時間
 * @param events2 グループ2のイベント発生フラグ
 * @return Log-rank検定の結果
 * @throws std::invalid_argument timesとeventsのサイズが一致しない場合、またはデータが空の場合
 */
inline logrank_result logrank_test(
    const std::vector<double>& times1,
    const std::vector<bool>& events1,
    const std::vector<double>& times2,
    const std::vector<bool>& events2)
{
    if (times1.size() != events1.size() || times2.size() != events2.size()) {
        throw std::invalid_argument("statcpp::logrank_test: times and events must have same length");
    }
    if (times1.empty() || times2.empty()) {
        throw std::invalid_argument("statcpp::logrank_test: empty data");
    }

    // すべてのユニークな時刻を取得してソート
    std::vector<double> all_times;
    all_times.reserve(times1.size() + times2.size());
    for (double t : times1) all_times.push_back(t);
    for (double t : times2) all_times.push_back(t);
    std::sort(all_times.begin(), all_times.end());
    all_times.erase(std::unique(all_times.begin(), all_times.end()), all_times.end());

    // リスク集合とイベント数を時刻ごとに計算
    std::size_t O1 = 0;  // グループ1の観測イベント数
    std::size_t O2 = 0;  // グループ2の観測イベント数
    double E1 = 0.0;     // グループ1の期待イベント数
    double var = 0.0;    // 分散

    // 各時刻でイベントが発生したか
    for (double t : all_times) {
        // この時刻より大きい（または等しい）観測数 = リスク集合
        std::size_t n1_risk = 0;
        std::size_t n2_risk = 0;
        std::size_t d1 = 0;  // グループ1のイベント数
        std::size_t d2 = 0;  // グループ2のイベント数

        for (std::size_t i = 0; i < times1.size(); ++i) {
            if (times1[i] >= t) {
                n1_risk++;
            }
            if (times1[i] == t && events1[i]) {
                d1++;
            }
        }

        for (std::size_t i = 0; i < times2.size(); ++i) {
            if (times2[i] >= t) {
                n2_risk++;
            }
            if (times2[i] == t && events2[i]) {
                d2++;
            }
        }

        std::size_t d = d1 + d2;  // 全イベント数
        std::size_t n_risk = n1_risk + n2_risk;  // 全リスク数

        if (d > 0 && n_risk > 0) {
            O1 += d1;
            O2 += d2;

            double e1 = static_cast<double>(n1_risk) * static_cast<double>(d) /
                       static_cast<double>(n_risk);
            E1 += e1;

            // 分散（超幾何分布の分散）
            if (n_risk > 1) {
                var += static_cast<double>(n1_risk) * static_cast<double>(n2_risk) *
                       static_cast<double>(d) * static_cast<double>(n_risk - d) /
                       (static_cast<double>(n_risk) * static_cast<double>(n_risk) *
                        static_cast<double>(n_risk - 1));
            }
        }
    }

    double E2 = static_cast<double>(O1 + O2) - E1;

    // 検定統計量
    double stat = 0.0;
    if (var > 0) {
        double diff = static_cast<double>(O1) - E1;
        stat = (diff * diff) / var;
    }

    // p値（χ²分布、自由度1）
    double p_value = 1.0 - statcpp::chisq_cdf(stat, 1.0);

    return {stat, p_value, 1, E1, E2, O1, O2};
}

// ============================================================================
// Median Survival Time
// ============================================================================

/**
 * @brief 中央生存時間を計算
 *
 * 生存確率が50%となる時刻を返します。
 * 50%に達しない場合はNaNを返します。
 *
 * @param km Kaplan-Meier推定の結果
 * @return 中央生存時間（50%に達しない場合はNaN）
 */
inline double median_survival_time(const kaplan_meier_result& km)
{
    // S(t) が最初に 0.5 以下になる時刻を探す
    for (std::size_t i = 0; i < km.survival.size(); ++i) {
        if (km.survival[i] <= 0.5) {
            return km.times[i];
        }
    }
    // 50% に達しない場合は NaN を返す
    return std::numeric_limits<double>::quiet_NaN();
}

// ============================================================================
// Hazard Rate (Actuarial Method)
// ============================================================================

/**
 * @brief ハザード率の結果
 *
 * 各時点でのハザード率と累積ハザードを保持します。
 */
struct hazard_rate_result {
    std::vector<double> times;          ///< 区間の開始時刻
    std::vector<double> hazard;         ///< ハザード率
    std::vector<double> cumulative_hazard;  ///< 累積ハザード
};

/**
 * @brief Nelson-Aalen 累積ハザード推定
 *
 * Nelson-Aalen推定量を用いて累積ハザード関数 H(t) を推定します。
 * 打ち切りデータを含む生存時間データに対応しています。
 *
 * @note Nelson-Aalen推定量は累積ハザード関数のノンパラメトリック推定量です：
 *       H(t) = Σ_{t_i ≤ t} d_i / n_i
 *       ここで d_i は時点 t_i でのイベント数、n_i はリスク集合のサイズです。
 *
 *       累積ハザードと生存関数の関係：
 *       - S(t) = exp(-H(t)) （指数変換による近似）
 *       - Kaplan-Meier推定量は S(t) = Π(1 - d_i/n_i) を直接推定
 *       - 小さいイベント確率では両者はほぼ一致
 *
 *       Nelson-Aalen推定の利点：
 *       - 信頼区間の構成がより単純（分散は Σd_i/n_i² で推定）
 *       - Cox回帰のベースラインハザード推定（Breslow推定量）と整合的
 *
 * @param times 観測時間のベクトル
 * @param events イベント発生フラグ（true = イベント発生、false = 打ち切り）
 * @return ハザード率の推定結果
 * @throws std::invalid_argument timesとeventsのサイズが異なる場合、またはデータが空の場合
 */
inline hazard_rate_result nelson_aalen(
    const std::vector<double>& times,
    const std::vector<bool>& events)
{
    if (times.size() != events.size()) {
        throw std::invalid_argument("statcpp::nelson_aalen: times and events must have same length");
    }
    if (times.empty()) {
        throw std::invalid_argument("statcpp::nelson_aalen: empty data");
    }

    std::size_t n = times.size();

    // インデックスを時間でソート
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&times](std::size_t i, std::size_t j) {
                  return times[i] < times[j];
              });

    hazard_rate_result result;
    result.times.push_back(0.0);
    result.hazard.push_back(0.0);
    result.cumulative_hazard.push_back(0.0);

    double H = 0.0;  // 累積ハザード
    std::size_t n_risk = n;

    std::size_t i = 0;
    while (i < n) {
        double t = times[indices[i]];
        std::size_t d = 0;
        std::size_t c = 0;

        while (i < n && times[indices[i]] == t) {
            if (events[indices[i]]) {
                d++;
            } else {
                c++;
            }
            i++;
        }

        if (d > 0 && n_risk > 0) {
            double h = static_cast<double>(d) / static_cast<double>(n_risk);
            H += h;

            result.times.push_back(t);
            result.hazard.push_back(h);
            result.cumulative_hazard.push_back(H);
        }

        n_risk -= (d + c);
    }

    return result;
}

} // namespace statcpp
