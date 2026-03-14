/**
 * @file example_random_engine.cpp
 * @brief 乱数エンジンのサンプルコード
 *
 * statcppライブラリの乱数エンジン管理機能（シード設定、
 * スレッドローカルエンジン、再現可能な乱数生成）の使用例を示します。
 */

#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <map>
#include "statcpp/random_engine.hpp"

int main() {
    std::cout << "=== 乱数エンジンの例 ===" << std::endl;

    // ============================================================================
    // 1. デフォルト乱数エンジンの使用
    // ============================================================================
    std::cout << "\n1. デフォルト乱数エンジンの使用" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::cout << "デフォルトエンジンの型: Mersenne Twister (mt19937_64)" << std::endl;

    // デフォルトエンジンから一様乱数を生成
    auto& engine = statcpp::get_random_engine();
    std::cout << "\nエンジンから10個の乱数を生成:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "  " << (i + 1) << ": " << engine() << std::endl;
    }

    // ============================================================================
    // 2. シードの設定
    // ============================================================================
    std::cout << "\n2. シードの設定と使用" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    // シードを固定して再現性を確保
    statcpp::set_seed(42);
    std::cout << "シードを42に設定" << std::endl;

    std::cout << "\n最初のシーケンス (seed=42):" << std::endl;
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (int i = 0; i < 5; ++i) {
        std::cout << "  " << std::fixed << std::setprecision(6) << uniform(engine) << std::endl;
    }

    // 同じシードで再度生成
    statcpp::set_seed(42);
    std::cout << "\n2番目のシーケンス (seed=42 再設定):" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "  " << std::fixed << std::setprecision(6) << uniform(engine) << std::endl;
    }

    std::cout << "\n注意: 同じシードのため、両方のシーケンスは同一です。" << std::endl;

    // ============================================================================
    // 3. 異なるシードでの生成
    // ============================================================================
    std::cout << "\n3. 異なるシードは異なるシーケンスを生成" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    statcpp::set_seed(100);
    std::cout << "seed=100でのシーケンス:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "  " << std::fixed << std::setprecision(6) << uniform(engine) << std::endl;
    }

    statcpp::set_seed(200);
    std::cout << "\nseed=200でのシーケンス:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "  " << std::fixed << std::setprecision(6) << uniform(engine) << std::endl;
    }

    // ============================================================================
    // 4. ランダムシードの使用
    // ============================================================================
    std::cout << "\n4. シードのランダム化" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    statcpp::randomize_seed();
    std::cout << "シードをランダム化しました（再現不可能）:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "  " << std::fixed << std::setprecision(6) << uniform(engine) << std::endl;
    }

    // ============================================================================
    // 5. 様々な分布での使用
    // ============================================================================
    std::cout << "\n5. 様々な分布での乱数エンジンの使用" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    statcpp::set_seed(42);  // 再現性のためシード設定

    // 一様分布 [0, 10)
    std::cout << "\n一様分布 [0, 10):" << std::endl;
    std::uniform_real_distribution<double> uniform_10(0.0, 10.0);
    for (int i = 0; i < 5; ++i) {
        std::cout << "  " << std::fixed << std::setprecision(4) << uniform_10(engine) << std::endl;
    }

    // 正規分布 N(0, 1)
    std::cout << "\n正規分布 N(0, 1):" << std::endl;
    std::normal_distribution<double> normal(0.0, 1.0);
    for (int i = 0; i < 5; ++i) {
        std::cout << "  " << std::fixed << std::setprecision(4) << normal(engine) << std::endl;
    }

    // 整数の一様分布 [1, 6] (サイコロ)
    std::cout << "\n整数の一様分布 [1, 6] (サイコロ):" << std::endl;
    std::uniform_int_distribution<int> dice(1, 6);
    for (int i = 0; i < 10; ++i) {
        std::cout << "  振り " << (i + 1) << ": " << dice(engine) << std::endl;
    }

    // ベルヌーイ分布（コイントス）
    std::cout << "\nベルヌーイ分布 (コイン投げ, p=0.5):" << std::endl;
    std::bernoulli_distribution coin(0.5);
    int heads = 0, tails = 0;
    for (int i = 0; i < 100; ++i) {
        if (coin(engine)) {
            ++heads;
        } else {
            ++tails;
        }
    }
    std::cout << "  100回の投げ: 表 = " << heads << ", 裏 = " << tails << std::endl;

    // ============================================================================
    // 6. 異なる乱数エンジンの比較
    // ============================================================================
    std::cout << "\n6. 異なる乱数エンジンの比較" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    // mt19937 (32-bit Mersenne Twister)
    std::mt19937 mt32(42);
    std::cout << "mt19937 (32-bit): ";
    for (int i = 0; i < 3; ++i) {
        std::cout << mt32() << " ";
    }
    std::cout << std::endl;

    // mt19937_64 (64-bit Mersenne Twister)
    std::mt19937_64 mt64(42);
    std::cout << "mt19937_64 (64-bit): ";
    for (int i = 0; i < 3; ++i) {
        std::cout << mt64() << " ";
    }
    std::cout << std::endl;

    // minstd_rand (Linear Congruential)
    std::minstd_rand lcg(42);
    std::cout << "minstd_rand (LCG): ";
    for (int i = 0; i < 3; ++i) {
        std::cout << lcg() << " ";
    }
    std::cout << std::endl;

    // ============================================================================
    // 7. 乱数の品質テスト（簡易版）
    // ============================================================================
    std::cout << "\n7. 簡易乱数品質テスト" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    statcpp::set_seed(12345);

    // [0, 1)の一様乱数を大量生成して統計を取る
    const int n_samples = 100000;
    std::vector<double> samples;
    samples.reserve(n_samples);

    std::uniform_real_distribution<double> unit_uniform(0.0, 1.0);
    for (int i = 0; i < n_samples; ++i) {
        samples.push_back(unit_uniform(engine));
    }

    // 平均を計算（理論値: 0.5）
    double sum = 0.0;
    for (double x : samples) {
        sum += x;
    }
    double mean = sum / n_samples;

    // 分散を計算（理論値: 1/12 ≈ 0.0833）
    double sum_sq = 0.0;
    for (double x : samples) {
        double diff = x - mean;
        sum_sq += diff * diff;
    }
    double variance = sum_sq / n_samples;

    std::cout << n_samples << "個の一様分布 [0,1) サンプルを生成:" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  平均 (期待値 0.5):      " << mean << std::endl;
    std::cout << "  分散 (期待値 0.083): " << variance << std::endl;

    // 分布の均一性チェック（10区間に分割）
    std::cout << "\n分布の均一性 (10区間):" << std::endl;
    std::map<int, int> bins;
    for (double x : samples) {
        int bin = static_cast<int>(x * 10.0);
        if (bin >= 10) bin = 9;  // 1.0のケース
        ++bins[bin];
    }

    for (int i = 0; i < 10; ++i) {
        double ratio = static_cast<double>(bins[i]) / n_samples;
        std::cout << "  [" << std::setw(3) << (i * 10) << "%-" << std::setw(3) << ((i + 1) * 10) << "%): "
                  << std::setw(6) << bins[i] << " (" << std::setw(5) << std::setprecision(2)
                  << (ratio * 100.0) << "%)" << std::endl;
    }
    std::cout << "  期待値: 各区間約10%" << std::endl;

    // ============================================================================
    // 8. スレッドセーフティの確認
    // ============================================================================
    std::cout << "\n8. スレッドローカルストレージ" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::cout << "乱数エンジンはスレッド安全性のためスレッドローカルです。" << std::endl;
    std::cout << "各スレッドは独立した乱数エンジンインスタンスを持ちます。" << std::endl;

    // ============================================================================
    // 9. 型トレイトの使用
    // ============================================================================
    std::cout << "\n9. 乱数エンジンの型トレイト" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::cout << "型トレイトのチェック:" << std::endl;
    std::cout << "  std::mt19937 は乱数エンジン: "
              << (statcpp::is_random_engine_v<std::mt19937> ? "はい" : "いいえ") << std::endl;
    std::cout << "  std::mt19937_64 は乱数エンジン: "
              << (statcpp::is_random_engine_v<std::mt19937_64> ? "はい" : "いいえ") << std::endl;
    std::cout << "  std::minstd_rand は乱数エンジン: "
              << (statcpp::is_random_engine_v<std::minstd_rand> ? "はい" : "いいえ") << std::endl;
    std::cout << "  int は乱数エンジン: "
              << (statcpp::is_random_engine_v<int> ? "はい" : "いいえ") << std::endl;

    // ============================================================================
    // 10. 実用例：モンテカルロ法で円周率を推定
    // ============================================================================
    std::cout << "\n10. 実用例：モンテカルロ法で円周率を推定" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    statcpp::set_seed(42);

    const int n_trials = 1000000;
    int inside_circle = 0;

    std::uniform_real_distribution<double> coord(-1.0, 1.0);

    for (int i = 0; i < n_trials; ++i) {
        double x = coord(engine);
        double y = coord(engine);
        if (x * x + y * y <= 1.0) {
            ++inside_circle;
        }
    }

    double pi_estimate = 4.0 * static_cast<double>(inside_circle) / n_trials;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "試行回数: " << n_trials << std::endl;
    std::cout << "円内の点の数: " << inside_circle << std::endl;
    std::cout << "推定π: " << pi_estimate << std::endl;
    std::cout << "実際のπ:    " << 3.141593 << std::endl;
    std::cout << "誤差:       " << std::abs(pi_estimate - 3.141593) << std::endl;

    std::cout << "\n=== 例の実行が正常に完了しました ===" << std::endl;

    return 0;
}
