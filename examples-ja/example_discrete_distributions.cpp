/**
 * @file example_discrete_distributions.cpp
 * @brief statcpp::discrete_distributions.hpp のサンプルコード
 *
 * このファイルでは、discrete_distributions.hpp で提供される
 * 離散確率分布の関数の使い方を実践的な例を通じて説明します。
 *
 * 【提供される分布】
 * - Binomial Distribution    : 二項分布
 * - Poisson Distribution     : ポアソン分布
 * - Bernoulli Distribution   : ベルヌーイ分布
 * - Discrete Uniform         : 離散一様分布
 * - Geometric Distribution   : 幾何分布
 *
 * 【コンパイル方法】
 * g++ -std=c++17 -I/path/to/statcpp/include example_discrete_distributions.cpp -o example_discrete_distributions
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include "statcpp/discrete_distributions.hpp"

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
// 1. 二項分布 (Binomial Distribution)
// ============================================================================

/**
 * @brief 二項分布の使用例
 *
 * 【概念】
 * n 回の独立な試行において、成功確率 p で成功する回数の分布
 *
 * 【数式】
 * P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
 *
 * 【パラメータ】
 * - n: 試行回数（正の整数）
 * - p: 各試行の成功確率（0 ≤ p ≤ 1）
 * - k: 成功回数（0 ≤ k ≤ n）
 *
 * 【使用場面】
 * - コイン投げ（n回投げて表が k回出る確率）
 * - 品質管理（n個のうち不良品が k個ある確率）
 * - A/B テスト（n人のユーザーのうち k人がクリックする確率）
 * - 選挙予測（n人の投票者のうち k人が特定候補に投票）
 */
void example_binomial() {
    print_section("1. 二項分布 (Binomial Distribution)");

    std::cout << R"(
【概念】
n 回の独立な試行において、成功確率 p で成功する回数の分布

【実例: コイン投げ】
公正なコイン（表の出る確率 p=0.5）を 10回投げる
→ 表が何回出るか？
)";

    int n = 10;
    double p = 0.5;

    print_subsection("確率質量関数 (PMF)");
    std::cout << "P(X=5 | n=10, p=0.5) = " << statcpp::binomial_pmf(5, n, p) << "\n";
    std::cout << "→ 10回中ちょうど5回表が出る確率\n";

    print_subsection("累積分布関数 (CDF)");
    std::cout << "P(X≤5 | n=10, p=0.5) = " << statcpp::binomial_cdf(5, n, p) << "\n";
    std::cout << "→ 10回中5回以下表が出る確率\n";

    print_subsection("分位点 (Quantile)");
    int median = statcpp::binomial_quantile(0.5, n, p);
    std::cout << "中央値 (0.5分位点) = " << median << "\n";
    std::cout << "→ 確率が50%を超える最小の成功回数\n";

    print_subsection("乱数生成");
    statcpp::set_seed(42);
    std::cout << "ランダムサンプル: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << statcpp::binomial_rand(n, p) << " ";
    }
    std::cout << "\n→ 10回投げたときの表の回数をシミュレーション\n";

    print_subsection("実用例: 品質管理");
    std::cout << R"(
製造ラインで1000個の製品を生産。不良率は2%。
100個サンプリングしたとき、不良品が3個以上含まれる確率は？
)";
    int sample_size = 100;
    double defect_rate = 0.02;
    double prob_3_or_more = 1.0 - statcpp::binomial_cdf(2, sample_size, defect_rate);
    std::cout << "P(X≥3) = 1 - P(X≤2) = " << prob_3_or_more << "\n";
    std::cout << "→ 約" << prob_3_or_more * 100 << "%の確率で3個以上の不良品\n";
}

// ============================================================================
// 2. ポアソン分布 (Poisson Distribution)
// ============================================================================

/**
 * @brief ポアソン分布の使用例
 *
 * 【概念】
 * 一定期間・一定領域内で発生するランダムな事象の回数の分布
 *
 * 【数式】
 * P(X = k) = (λ^k × e^(-λ)) / k!
 *
 * 【パラメータ】
 * - λ (lambda): 平均発生回数（λ > 0）
 *
 * 【使用場面】
 * - 1時間あたりの来客数
 * - 1日あたりのシステム障害発生回数
 * - Webサイトの1分あたりのアクセス数
 * - 自然災害の年間発生回数
 */
void example_poisson() {
    print_section("2. ポアソン分布 (Poisson Distribution)");

    std::cout << R"(
【概念】
一定期間・一定領域内で発生するランダムな事象の回数の分布

【実例: コールセンター】
コールセンターに1時間平均3件の問い合わせがある
→ 次の1時間に何件来るか？
)";

    double lambda = 3.0;

    print_subsection("確率質量関数 (PMF)");
    std::cout << "P(X=3 | λ=3.0) = " << statcpp::poisson_pmf(3, lambda) << "\n";
    std::cout << "→ ちょうど3件来る確率\n";

    std::cout << "\nP(X=0 | λ=3.0) = " << statcpp::poisson_pmf(0, lambda) << "\n";
    std::cout << "→ 1件も来ない確率\n";

    print_subsection("累積分布関数 (CDF)");
    std::cout << "P(X≤5 | λ=3.0) = " << statcpp::poisson_cdf(5, lambda) << "\n";
    std::cout << "→ 5件以下である確率\n";

    print_subsection("分位点 (Quantile)");
    int p95 = statcpp::poisson_quantile(0.95, lambda);
    std::cout << "95パーセンタイル = " << p95 << "\n";
    std::cout << "→ 95%の確率でこの件数以下\n";

    print_subsection("乱数生成");
    statcpp::set_seed(42);
    std::cout << "1時間ごとの問い合わせ件数（5時間分）: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << statcpp::poisson_rand(lambda) << " ";
    }
    std::cout << "\n";

    print_subsection("実用例: サーバー負荷予測");
    std::cout << R"(
Webサーバーに1分平均10件のリクエスト（λ=10）
1分間に15件以上のリクエストが来る確率は？
)";
    double server_lambda = 10.0;
    double prob_15_or_more = 1.0 - statcpp::poisson_cdf(14, server_lambda);
    std::cout << "P(X≥15) = 1 - P(X≤14) = " << prob_15_or_more << "\n";
    std::cout << "→ 約" << prob_15_or_more * 100 << "%の確率で高負荷状態\n";
}

// ============================================================================
// 3. ベルヌーイ分布 (Bernoulli Distribution)
// ============================================================================

/**
 * @brief ベルヌーイ分布の使用例
 *
 * 【概念】
 * 1回の試行で成功（1）または失敗（0）のみが起こる分布
 * 二項分布で n=1 の特殊ケース
 *
 * 【数式】
 * P(X = 1) = p
 * P(X = 0) = 1-p
 *
 * 【パラメータ】
 * - p: 成功確率（0 ≤ p ≤ 1）
 *
 * 【使用場面】
 * - コイン1回投げ
 * - ユーザーがボタンをクリックするか/しないか
 * - 製品1個が良品か/不良品か
 * - Yes/No の二択
 */
void example_bernoulli() {
    print_section("3. ベルヌーイ分布 (Bernoulli Distribution)");

    std::cout << R"(
【概念】
1回の試行で成功（1）または失敗（0）のみが起こる分布

【実例: 広告クリック】
広告のクリック率が70%（p=0.7）
→ 次のユーザーがクリックするか？
)";

    double p = 0.7;

    print_subsection("確率質量関数 (PMF)");
    std::cout << "P(X=1 | p=0.7) = " << statcpp::bernoulli_pmf(1, p) << " (クリックする)\n";
    std::cout << "P(X=0 | p=0.7) = " << statcpp::bernoulli_pmf(0, p) << " (クリックしない)\n";

    print_subsection("累積分布関数 (CDF)");
    std::cout << "P(X≤0 | p=0.7) = " << statcpp::bernoulli_cdf(0, p) << "\n";
    std::cout << "→ クリックしない確率\n";

    print_subsection("分位点 (Quantile)");
    int median = statcpp::bernoulli_quantile(0.5, p);
    std::cout << "中央値 (0.5分位点) = " << median << "\n";

    print_subsection("乱数生成");
    statcpp::set_seed(42);
    std::cout << "10ユーザーのクリック結果（1=クリック, 0=非クリック）:\n   ";
    int click_count = 0;
    for (int i = 0; i < 10; ++i) {
        int result = statcpp::bernoulli_rand(p);
        std::cout << result << " ";
        click_count += result;
    }
    std::cout << "\n→ 10人中" << click_count << "人がクリック\n";
}

// ============================================================================
// 4. 離散一様分布 (Discrete Uniform Distribution)
// ============================================================================

/**
 * @brief 離散一様分布の使用例
 *
 * 【概念】
 * a から b までの整数がすべて等しい確率で出現する分布
 *
 * 【数式】
 * P(X = k) = 1 / (b - a + 1)  (a ≤ k ≤ b)
 *
 * 【パラメータ】
 * - a: 最小値（整数）
 * - b: 最大値（整数、a ≤ b）
 *
 * 【使用場面】
 * - サイコロを振る（1〜6）
 * - ランダムな整数を選ぶ
 * - くじ引き
 * - ランダムサンプリング
 */
void example_discrete_uniform() {
    print_section("4. 離散一様分布 (Discrete Uniform Distribution)");

    std::cout << R"(
【概念】
すべての値が等しい確率で出現する分布

【実例: サイコロ】
公正な6面サイコロ（1〜6が等確率）
)";

    int a = 1, b = 6;

    print_subsection("確率質量関数 (PMF)");
    std::cout << "P(X=3 | a=1, b=6) = " << statcpp::discrete_uniform_pmf(3, a, b) << "\n";
    std::cout << "→ すべての目が等確率 (1/6 = " << 1.0/6.0 << ")\n";

    print_subsection("累積分布関数 (CDF)");
    std::cout << "P(X≤3 | a=1, b=6) = " << statcpp::discrete_uniform_cdf(3, a, b) << "\n";
    std::cout << "→ 3以下の目が出る確率 (3/6 = 0.5)\n";

    print_subsection("分位点 (Quantile)");
    int median = statcpp::discrete_uniform_quantile(0.5, a, b);
    std::cout << "中央値 (0.5分位点) = " << median << "\n";

    print_subsection("乱数生成");
    statcpp::set_seed(42);
    std::cout << "サイコロを10回振った結果: ";
    std::vector<int> counts(7, 0);
    for (int i = 0; i < 10; ++i) {
        int result = statcpp::discrete_uniform_rand(a, b);
        std::cout << result << " ";
        counts[result]++;
    }
    std::cout << "\n\n度数分布:\n";
    for (int i = 1; i <= 6; ++i) {
        std::cout << "   " << i << "の目: " << counts[i] << "回\n";
    }
}

// ============================================================================
// 5. 幾何分布 (Geometric Distribution)
// ============================================================================

/**
 * @brief 幾何分布の使用例
 *
 * 【概念】
 * 初めて成功するまでに必要な試行回数の分布
 *
 * 【数式】
 * P(X = k) = (1-p)^(k-1) × p
 *
 * 【パラメータ】
 * - p: 各試行の成功確率（0 < p ≤ 1）
 * - k: 試行回数（k ≥ 0 or k ≥ 1、実装により異なる）
 *
 * 【使用場面】
 * - 初めて成功するまでの試行回数
 * - 機械が故障するまでの使用回数
 * - 顧客が初めて購入するまでの訪問回数
 * - システムエラーが発生するまでの時間
 *
 * 【無記憶性】
 * 幾何分布は「無記憶性」を持つ：
 * これまで失敗していても、次に成功する確率は変わらない
 */
void example_geometric() {
    print_section("5. 幾何分布 (Geometric Distribution)");

    std::cout << R"(
【概念】
初めて成功するまでに必要な試行回数の分布

【実例: バスケットボールのフリースロー】
成功確率 p=0.3 のシューターが、
初めて成功するまで何回投げる必要があるか？
)";

    double p = 0.3;

    print_subsection("確率質量関数 (PMF)");
    std::cout << "P(X=2 | p=0.3) = " << statcpp::geometric_pmf(2, p) << "\n";
    std::cout << "→ 2回目で初めて成功する確率\n";
    std::cout << "   （1回目失敗、2回目成功）\n";

    std::cout << "\nP(X=1 | p=0.3) = " << statcpp::geometric_pmf(1, p) << "\n";
    std::cout << "→ 1回目で成功する確率\n";

    print_subsection("累積分布関数 (CDF)");
    std::cout << "P(X≤5 | p=0.3) = " << statcpp::geometric_cdf(5, p) << "\n";
    std::cout << "→ 5回以内に成功する確率\n";

    print_subsection("分位点 (Quantile)");
    int median = statcpp::geometric_quantile(0.5, p);
    std::cout << "中央値 (0.5分位点) = " << median << "\n";
    std::cout << "→ 50%の確率でこの回数以内に成功\n";

    print_subsection("乱数生成");
    statcpp::set_seed(42);
    std::cout << "5回のシミュレーション（初成功までの試行回数）: ";
    double total = 0;
    for (int i = 0; i < 5; ++i) {
        int trials = statcpp::geometric_rand(p);
        std::cout << trials << " ";
        total += trials;
    }
    std::cout << "\n平均: " << total / 5.0 << "回\n";
    std::cout << "理論的期待値: " << 1.0 / p << "回\n";

    print_subsection("実用例: カスタマーサポート");
    std::cout << R"(
電話がつながる確率が20%（p=0.2）のサポートセンター
10回以内につながる確率は？
)";
    double support_p = 0.2;
    double prob_within_10 = statcpp::geometric_cdf(10, support_p);
    std::cout << "P(X≤10) = " << prob_within_10 << "\n";
    std::cout << "→ 約" << prob_within_10 * 100 << "%の確率で10回以内につながる\n";
}

// ============================================================================
// 6. 分布の比較と使い分け
// ============================================================================

void example_comparison() {
    print_section("6. 離散分布の比較と使い分け");

    std::cout << R"(
┌─────────────────┬────────────────────────────────────────────────┐
│ 分布            │ 使用場面                                       │
├─────────────────┼────────────────────────────────────────────────┤
│ Bernoulli       │ 1回の試行（成功/失敗）                         │
│ (ベルヌーイ)    │ 例: 1回のコイン投げ、1人のクリック判定        │
├─────────────────┼────────────────────────────────────────────────┤
│ Binomial        │ n回の独立試行での成功回数                      │
│ (二項)          │ 例: n回のコイン投げで表が出る回数             │
├─────────────────┼────────────────────────────────────────────────┤
│ Geometric       │ 初めて成功するまでの試行回数                   │
│ (幾何)          │ 例: 初めて表が出るまでコインを投げる回数       │
├─────────────────┼────────────────────────────────────────────────┤
│ Poisson         │ 一定期間/領域でのランダムな事象の発生回数     │
│ (ポアソン)      │ 例: 1時間あたりの来客数、1日のエラー回数      │
├─────────────────┼────────────────────────────────────────────────┤
│ Discrete        │ すべての値が等確率                             │
│ Uniform         │ 例: サイコロ、くじ引き                         │
│ (離散一様)      │                                                │
└─────────────────┴────────────────────────────────────────────────┘

【Binomial vs Poisson の使い分け】

Binomial を使う場合:
- 試行回数 n が明確
- 各試行の成功確率 p が既知
- n が小さい〜中程度

Poisson を使う場合:
- 試行回数が非常に大きい（n → ∞）
- 成功確率が非常に小さい（p → 0）
- np = λ が一定
- 「稀な事象」のモデル化

【近似関係】
n が大きく p が小さいとき、Binomial(n, p) ≈ Poisson(λ=np)
)";

    print_subsection("実例: Binomial → Poisson 近似");
    int n = 1000;
    double p = 0.003;  // 小さい確率
    double lambda = n * p;  // λ = 3.0

    int k = 5;
    double binomial_prob = statcpp::binomial_pmf(k, n, p);
    double poisson_prob = statcpp::poisson_pmf(k, lambda);

    std::cout << "n=1000, p=0.003, k=5 のとき:\n";
    std::cout << "  Binomial PMF: " << binomial_prob << "\n";
    std::cout << "  Poisson PMF (λ=3):  " << poisson_prob << "\n";
    std::cout << "  相対誤差: " << std::abs(binomial_prob - poisson_prob) / binomial_prob * 100 << "%\n";
    std::cout << "\n→ n が大きく p が小さい場合、Poisson が良い近似\n";
}

// ============================================================================
// まとめ
// ============================================================================

void print_summary() {
    print_section("まとめ：離散分布の関数");

    std::cout << R"(
【各分布で共通する関数】
- XXX_pmf(k, params...)      : 確率質量関数 P(X=k)
- XXX_cdf(k, params...)      : 累積分布関数 P(X≤k)
- XXX_quantile(p, params...) : 分位点（逆CDF）
- XXX_rand(params...)        : 乱数生成

【重要な概念】
1. PMF (確率質量関数): 特定の値kになる確率
2. CDF (累積分布関数): k以下になる確率の累積
3. Quantile (分位点): 指定された確率pに対応するk値
4. 期待値 E[X] と分散 Var[X] は分布によって異なる

【実用上のヒント】
- 小規模データ・明確な試行回数 → Binomial
- 大規模・稀な事象 → Poisson
- 初成功までの回数 → Geometric
- 等確率の離散選択 → Discrete Uniform
- 1回の試行 → Bernoulli

【統計的推論への応用】
これらの分布は、推定・検定の基礎となります：
- 二項検定（Binomial test）
- ポアソン検定（Poisson test）
- カイ二乗適合度検定

詳細は estimation.hpp, parametric_tests.hpp を参照してください。
)";
}

// ============================================================================
// メイン関数
// ============================================================================

int main()
{
    std::cout << std::fixed << std::setprecision(4);

    // 各サンプルを実行
    example_binomial();
    example_poisson();
    example_bernoulli();
    example_discrete_uniform();
    example_geometric();
    example_comparison();

    // まとめを表示
    print_summary();

    return 0;
}
