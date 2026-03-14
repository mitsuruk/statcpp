/**
 * @file example_numerical_utils.cpp
 * @brief statcpp::numerical_utils.hpp のサンプルコード
 *
 * このファイルでは、numerical_utils.hpp で提供される
 * 数値計算ユーティリティの関数の使い方を実践的な例を通じて説明します。
 *
 * 【提供される関数】
 * - approx_equal()            : 浮動小数点数の近似等価判定
 * - has_converged()           : 収束判定
 * - log1p_safe(), expm1_safe(): 安全な数学関数
 * - kahan_sum()               : 高精度加算（Kahan summation）
 * - safe_divide()             : 安全な除算（ゼロ除算回避）
 * - clamp()                   : 値の範囲制限
 * - in_range()                : 範囲内判定
 *
 * 【コンパイル方法】
 * g++ -std=c++17 -I/path/to/statcpp/include example_numerical_utils.cpp -o example_numerical_utils
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include "statcpp/numerical_utils.hpp"

// ============================================================================
// 結果表示用のヘルパー関数
// ============================================================================

void print_section(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

void print_subsection(const std::string& title) {
    std::cout << "\n--- " << title << " ---\n";
}

// ============================================================================
// 1. 浮動小数点数の問題と近似等価判定
// ============================================================================

/**
 * @brief 浮動小数点数の問題と近似等価判定の使用例
 *
 * 【問題】
 * コンピュータでは浮動小数点数を正確に表現できないため、
 * 数学的に等しいはずの計算結果が異なることがある
 *
 * 【原因】
 * - 2進数での表現限界
 * - 丸め誤差の累積
 * - 桁落ち
 *
 * 【解決策】
 * approx_equal() を使って「ほぼ等しい」を判定
 */
void example_floating_point_comparison() {
    print_section("1. 浮動小数点数の比較");

    std::cout << R"(
【問題: 浮動小数点数の丸め誤差】

数学的には 0.1 + 0.2 = 0.3 ですが、コンピュータでは...
)";

    std::cout << std::setprecision(17);  // 高精度表示
    double a = 0.1 + 0.2;
    double b = 0.3;

    std::cout << "0.1 + 0.2 の結果: " << a << "\n";
    std::cout << "0.3 の値:         " << b << "\n";
    std::cout << "差分:             " << (a - b) << "\n";

    print_subsection("通常の比較演算子（==）");
    std::cout << "0.1 + 0.2 == 0.3 ? " << (a == b ? "true" : "false") << "\n";
    std::cout << "→ false になってしまう！\n";

    print_subsection("approx_equal() を使った比較");
    std::cout << std::setprecision(4);
    bool approx = statcpp::approx_equal(a, b);
    std::cout << "approx_equal(0.1+0.2, 0.3) ? " << (approx ? "true" : "false") << "\n";
    std::cout << "→ 正しく true と判定される\n";

    print_subsection("実用例: ループの終了判定");
    std::cout << R"(
悪い例（== を使用）:
    double x = 0.0;
    while (x != 1.0) {
        x += 0.1;
    }
    // 無限ループになる可能性！

良い例（approx_equal を使用）:
    double x = 0.0;
    while (!approx_equal(x, 1.0)) {
        x += 0.1;
    }
    // 正しく終了する
)";

    std::cout << "\n実際にループを実行:\n";
    double x = 0.0;
    int count = 0;
    std::cout << std::setprecision(10);
    while (!statcpp::approx_equal(x, 1.0) && count < 15) {
        std::cout << "  x = " << x << "\n";
        x += 0.1;
        count++;
    }
    std::cout << "  最終的な x = " << x << "\n";
    std::cout << "  ループ回数: " << count << "\n";
    std::cout << std::setprecision(4);
}

// ============================================================================
// 2. 収束判定
// ============================================================================

/**
 * @brief 収束判定の使用例
 *
 * 【概念】
 * 反復計算で、値がほとんど変化しなくなったら計算を終了する
 *
 * 【使用場面】
 * - ニュートン法などの反復アルゴリズム
 * - 最適化アルゴリズム
 * - 不動点反復
 * - EM アルゴリズム
 */
void example_convergence() {
    print_section("2. 収束判定 (has_converged)");

    std::cout << R"(
【概念】
反復計算で、前回の値と今回の値がほとんど変わらなくなったら
「収束した」と判定して計算を終了

【実例: 平方根の計算（バビロニア法）】
√2 を反復計算で求める
)";

    print_subsection("反復計算の過程");
    double target = 2.0;
    double x = 1.0;  // 初期値
    const int max_iter = 20;

    std::cout << std::setprecision(10);
    std::cout << "  反復 0: x = " << x << "\n";

    for (int i = 1; i <= max_iter; ++i) {
        double x_new = 0.5 * (x + target / x);  // バビロニア法の更新式

        std::cout << "  反復 " << i << ": x = " << x_new;

        if (statcpp::has_converged(x_new, x)) {
            std::cout << " ← 収束！\n";
            std::cout << "\n真の値: " << std::sqrt(target) << "\n";
            std::cout << "計算値: " << x_new << "\n";
            std::cout << "誤差:   " << std::abs(x_new - std::sqrt(target)) << "\n";
            std::cout << "収束までの反復回数: " << i << "\n";
            break;
        } else {
            std::cout << "\n";
        }

        x = x_new;
    }
    std::cout << std::setprecision(4);

    print_subsection("実用上の注意");
    std::cout << R"(
収束判定を使わない場合:
- 不必要な計算を続けてしまう（計算時間の無駄）
- いつ終了すべきか判断できない

収束判定を使う場合:
- 十分な精度に達したら自動的に終了
- 計算効率が向上
)";
}

// ============================================================================
// 3. 高精度加算 (Kahan Summation)
// ============================================================================

/**
 * @brief Kahan 加算の使用例
 *
 * 【問題】
 * 大きな数と小さな数を加算すると、小さな数が消失する（桁落ち）
 *
 * 【解決策】
 * Kahan 加算アルゴリズムで丸め誤差を補正
 *
 * 【使用場面】
 * - 大量の数値の合計
 * - 桁数の差が大きい数値の加算
 * - 高精度が求められる統計計算
 */
void example_kahan_sum() {
    print_section("3. 高精度加算 (Kahan Summation)");

    std::cout << R"(
【問題: 桁落ち】

大きな数と小さな数を足すと、小さな数が「消える」

【実例】
10^10 + 1 + 1 - 10^10 を計算
数学的には 2 になるはずだが...
)";

    std::vector<double> values = {1e10, 1.0, 1.0, -1e10};

    print_subsection("通常の加算");
    double naive_sum = 0.0;
    for (double v : values) {
        naive_sum += v;
    }
    std::cout << "計算値: " << naive_sum << "\n";
    std::cout << "期待値: 2.0\n";
    std::cout << "誤差:   " << std::abs(naive_sum - 2.0) << "\n";
    std::cout << "→ 丸め誤差で正しい結果が得られない！\n";

    print_subsection("Kahan 加算");
    double kahan = statcpp::kahan_sum(values.begin(), values.end());
    std::cout << "計算値: " << kahan << "\n";
    std::cout << "期待値: 2.0\n";
    std::cout << "誤差:   " << std::abs(kahan - 2.0) << "\n";
    std::cout << "→ 正しい結果が得られる！\n";

    print_subsection("実用例: 大量データの合計");
    std::cout << R"(
1000個の 0.1 を足す（数学的には 100）
)";

    std::vector<double> many_small_values(1000, 0.1);

    double naive_many = 0.0;
    for (double v : many_small_values) {
        naive_many += v;
    }

    double kahan_many = statcpp::kahan_sum(many_small_values.begin(),
                                            many_small_values.end());

    std::cout << std::setprecision(10);
    std::cout << "通常の加算:   " << naive_many << "\n";
    std::cout << "Kahan 加算:   " << kahan_many << "\n";
    std::cout << "期待値:       100.0\n";
    std::cout << "通常の誤差:   " << std::abs(naive_many - 100.0) << "\n";
    std::cout << "Kahan の誤差: " << std::abs(kahan_many - 100.0) << "\n";
    std::cout << std::setprecision(4);

    std::cout << "\n→ Kahan 加算の方が高精度\n";

    print_subsection("いつ Kahan 加算を使うべきか");
    std::cout << R"(
✅ 使うべき場合:
- 大量の数値を加算する
- 桁数の差が大きい数値を扱う
- 統計量（平均、分散など）の高精度計算
- 数値積分

❌ 不要な場合:
- 少数の数値の加算
- 桁数が近い数値のみを扱う
- 計算速度が最優先
)";
}

// ============================================================================
// 4. 安全な除算
// ============================================================================

/**
 * @brief 安全な除算の使用例
 *
 * 【問題】
 * ゼロ除算はプログラムをクラッシュさせる
 *
 * 【解決策】
 * safe_divide() でゼロ除算を検出し、デフォルト値を返す
 *
 * 【使用場面】
 * - ユーザー入力を使った除算
 * - データ分析（欠損値や異常値の可能性）
 * - 比率の計算
 */
void example_safe_division() {
    print_section("4. 安全な除算 (safe_divide)");

    std::cout << R"(
【問題: ゼロ除算】

通常の除算では、分母が 0 の場合にエラーが発生

【実例】
)";

    print_subsection("safe_divide() の使用");

    double a = 10.0;
    double b = 2.0;
    double c = 0.0;

    std::cout << "10 / 2 = " << statcpp::safe_divide(a, b) << "\n";
    std::cout << "→ 通常通り計算\n\n";

    std::cout << "10 / 0 (デフォルト: NaN) = " << statcpp::safe_divide(a, c) << "\n";
    std::cout << "→ NaN を返す（安全）\n\n";

    std::cout << "10 / 0 (デフォルト: 0) = " << statcpp::safe_divide(a, c, 0.0) << "\n";
    std::cout << "→ 指定したデフォルト値（0）を返す\n\n";

    std::cout << "10 / 0 (デフォルト: ∞) = "
              << statcpp::safe_divide(a, c, std::numeric_limits<double>::infinity()) << "\n";
    std::cout << "→ 無限大を返す\n";

    print_subsection("実用例: 比率の計算");
    std::cout << R"(
クリック率（CTR）の計算: クリック数 / 表示回数
)";

    std::vector<int> impressions = {1000, 500, 0, 200};  // 表示回数
    std::vector<int> clicks = {50, 30, 0, 15};           // クリック数

    std::cout << "\n広告キャンペーンのクリック率:\n";
    for (size_t i = 0; i < impressions.size(); ++i) {
        double ctr = statcpp::safe_divide(
            static_cast<double>(clicks[i]),
            static_cast<double>(impressions[i]),
            0.0  // 表示回数が 0 なら CTR も 0
        );
        std::cout << "  キャンペーン " << (i+1) << ": "
                  << clicks[i] << "/" << impressions[i]
                  << " = " << ctr * 100 << "%\n";
    }

    print_subsection("注意点");
    std::cout << R"(
デフォルト値の選択は用途によって異なる:
- 統計計算 → NaN（欠損値として扱う）
- ビジネスメトリクス → 0（該当データなし）
- 数学的な極限 → ∞（無限大）

用途に応じて適切なデフォルト値を選ぶことが重要！
)";
}

// ============================================================================
// 5. 値の範囲制限 (Clamp)
// ============================================================================

/**
 * @brief clamp() と in_range() の使用例
 *
 * 【概念】
 * clamp: 値を指定範囲内に収める
 * in_range: 値が範囲内かチェック
 *
 * 【使用場面】
 * - ユーザー入力の検証
 * - 確率値の範囲制限（0〜1）
 * - 画像処理（ピクセル値の範囲）
 * - パラメータの妥当性チェック
 */
void example_clamp() {
    print_section("5. 値の範囲制限 (clamp / in_range)");

    std::cout << R"(
【概念】
clamp(x, min, max): x を [min, max] の範囲に収める
in_range(x, min, max): x が [min, max] の範囲内か判定
)";

    print_subsection("clamp() の動作");
    std::cout << "範囲: [0, 10]\n";
    std::cout << "  clamp(5, 0, 10)  = " << statcpp::clamp(5.0, 0.0, 10.0) << " (範囲内)\n";
    std::cout << "  clamp(-5, 0, 10) = " << statcpp::clamp(-5.0, 0.0, 10.0) << " (下限に制限)\n";
    std::cout << "  clamp(15, 0, 10) = " << statcpp::clamp(15.0, 0.0, 10.0) << " (上限に制限)\n";

    print_subsection("in_range() の動作");
    std::cout << "範囲: [0, 10]\n";
    std::cout << "  in_range(5, 0, 10)  = " << (statcpp::in_range(5.0, 0.0, 10.0) ? "true" : "false") << "\n";
    std::cout << "  in_range(-5, 0, 10) = " << (statcpp::in_range(-5.0, 0.0, 10.0) ? "true" : "false") << "\n";
    std::cout << "  in_range(15, 0, 10) = " << (statcpp::in_range(15.0, 0.0, 10.0) ? "true" : "false") << "\n";

    print_subsection("実用例 1: 確率値の範囲制限");
    std::cout << R"(
確率は必ず [0, 1] の範囲でなければならない
)";

    std::vector<double> probabilities = {0.5, -0.1, 1.2, 0.0, 1.0};
    std::cout << "\n計算された確率値を [0, 1] に制限:\n";
    for (double p : probabilities) {
        double clamped = statcpp::clamp(p, 0.0, 1.0);
        std::cout << "  " << p << " → " << clamped;
        if (p != clamped) {
            std::cout << " (制限された)";
        }
        std::cout << "\n";
    }

    print_subsection("実用例 2: ユーザー入力の検証");
    std::cout << R"(
年齢の入力を検証（0〜120歳が妥当な範囲）
)";

    std::vector<int> ages = {25, -5, 150, 0, 120};
    std::cout << "\n入力された年齢の検証:\n";
    for (int age : ages) {
        bool valid = statcpp::in_range(static_cast<double>(age), 0.0, 120.0);
        std::cout << "  年齢 " << age << ": "
                  << (valid ? "✓ 妥当" : "✗ 不正") << "\n";
    }

    print_subsection("実用例 3: スコアの正規化");
    std::cout << R"(
テストスコアを 0〜100 の範囲に制限
)";

    std::vector<double> raw_scores = {85, 120, -10, 0, 100, 105};
    std::cout << "\n生スコア → 正規化スコア:\n";
    for (double score : raw_scores) {
        double normalized = statcpp::clamp(score, 0.0, 100.0);
        std::cout << "  " << score << " → " << normalized;
        if (score != normalized) {
            std::cout << " (範囲外を修正)";
        }
        std::cout << "\n";
    }
}

// ============================================================================
// 6. 安全な数学関数 (log1p, expm1)
// ============================================================================

/**
 * @brief log1p_safe() と expm1_safe() の使用例
 *
 * 【問題】
 * x が 0 に近いとき、log(1+x) や exp(x)-1 の計算精度が低下
 *
 * 【解決策】
 * log1p(x) = log(1+x) を高精度で計算
 * expm1(x) = exp(x)-1 を高精度で計算
 *
 * 【使用場面】
 * - 小さな変化率の計算
 * - 金融計算（複利計算）
 * - 統計計算（対数変換）
 */
void example_safe_math() {
    print_section("6. 安全な数学関数 (log1p / expm1)");

    std::cout << R"(
【問題: 桁落ち】

x が 0 に近いとき、log(1+x) や exp(x)-1 の精度が低下

【実例: 小さな利率の複利計算】
)";

    double small_x = 1e-10;

    print_subsection("log1p (log(1+x) の高精度版)");
    std::cout << std::setprecision(15);
    std::cout << "x = " << small_x << " のとき:\n";
    std::cout << "  log(1 + x)  = " << std::log(1.0 + small_x) << " (通常)\n";
    std::cout << "  log1p(x)    = " << statcpp::log1p_safe(small_x) << " (高精度)\n";
    std::cout << "  理論値      ≈ " << small_x << " (x が小さいとき)\n";

    print_subsection("expm1 (exp(x)-1 の高精度版)");
    std::cout << "x = " << small_x << " のとき:\n";
    std::cout << "  exp(x) - 1  = " << (std::exp(small_x) - 1.0) << " (通常)\n";
    std::cout << "  expm1(x)    = " << statcpp::expm1_safe(small_x) << " (高精度)\n";
    std::cout << "  理論値      ≈ " << small_x << " (x が小さいとき)\n";
    std::cout << std::setprecision(4);

    print_subsection("実用例: 複利計算");
    std::cout << R"(
年利 0.01%（= 0.0001）で 1年間運用した場合の利益
元本: 1,000,000 円
)";

    double principal = 1000000.0;
    double rate = 0.0001;  // 0.01%

    double interest_naive = principal * (std::exp(rate) - 1.0);
    double interest_accurate = principal * statcpp::expm1_safe(rate);

    std::cout << "\n通常の計算:   " << interest_naive << " 円\n";
    std::cout << "高精度計算:   " << interest_accurate << " 円\n";
    std::cout << "理論値:       " << principal * rate << " 円\n";
    std::cout << "\n→ 高精度計算の方が正確\n";
}

// ============================================================================
// まとめ
// ============================================================================

void print_summary() {
    print_section("まとめ：数値計算ユーティリティ");

    std::cout << R"(
┌─────────────────────┬──────────────────────────────────────────┐
│ 関数                │ 用途                                     │
├─────────────────────┼──────────────────────────────────────────┤
│ approx_equal()      │ 浮動小数点数の近似等価判定               │
│ has_converged()     │ 反復計算の収束判定                       │
│ kahan_sum()         │ 高精度加算（丸め誤差の軽減）             │
│ safe_divide()       │ ゼロ除算を回避する安全な除算             │
│ clamp()             │ 値を指定範囲に制限                       │
│ in_range()          │ 値が範囲内かチェック                     │
│ log1p_safe()        │ log(1+x) の高精度計算                    │
│ expm1_safe()        │ exp(x)-1 の高精度計算                    │
└─────────────────────┴──────────────────────────────────────────┘

【浮動小数点数の注意点】
1. == による比較は避ける → approx_equal() を使用
2. 大量の加算では丸め誤差が累積 → kahan_sum() を使用
3. 小さい値の log, exp 計算は精度低下 → log1p, expm1 を使用

【実用上のベストプラクティス】
✅ DO:
- 浮動小数点数の比較には approx_equal()
- 反復計算には収束判定を実装
- ユーザー入力値は clamp() で範囲制限
- ゼロ除算の可能性がある場合は safe_divide()

❌ DON'T:
- 浮動小数点数に == を使わない
- 無限ループのリスクを無視しない
- 丸め誤差を無視しない

【パフォーマンスとのトレードオフ】
- 高精度関数（Kahan加算など）は若干遅い
- 精度が重要な場合のみ使用
- 大量データや反復計算では効果大
)";
}

// ============================================================================
// メイン関数
// ============================================================================

int main()
{
    std::cout << std::fixed << std::setprecision(4);

    // 各サンプルを実行
    example_floating_point_comparison();
    example_convergence();
    example_kahan_sum();
    example_safe_division();
    example_clamp();
    example_safe_math();

    // まとめを表示
    print_summary();

    return 0;
}
