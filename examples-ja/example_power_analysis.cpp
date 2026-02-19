/**
 * @file example_power_analysis.cpp
 * @brief パワー解析（検出力分析）の包括的なサンプルコード
 *
 * このファイルでは、統計的検定におけるパワー解析の実践的な使用例を示します。
 * パワー解析は、研究設計において適切なサンプルサイズを決定したり、
 * 既存のデータの検出力を評価したりする際に不可欠なツールです。
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include "statcpp/power_analysis.hpp"
#include "statcpp/effect_size.hpp"

// ヘルパー関数：セクションヘッダーの表示
void print_section(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(70, '=') << "\n\n";
}

// ヘルパー関数：サブセクションヘッダーの表示
void print_subsection(const std::string& title) {
    std::cout << "\n" << std::string(60, '-') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(60, '-') << "\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // ============================================================
    // 1. パワー解析の基本概念
    // ============================================================
    print_section("1. パワー解析の基本概念");

    std::cout << "パワー解析では、以下の4つの要素を扱います：\n\n";
    std::cout << "1. サンプルサイズ (n): 必要な観測数\n";
    std::cout << "2. 効果量 (effect size): 検出したい効果の大きさ\n";
    std::cout << "   - Cohen's d: 標準化された平均差\n";
    std::cout << "   - 小: 0.2, 中: 0.5, 大: 0.8 (Cohen, 1988)\n";
    std::cout << "3. 有意水準 (α): 第1種過誤の確率（通常 0.05）\n";
    std::cout << "4. 検出力 (1-β): 真の効果を検出できる確率（通常 0.80 以上を目指す）\n\n";

    std::cout << "これら4つの要素のうち、3つが決まれば残りの1つを計算できます。\n";
    std::cout << "最も一般的な用途は、効果量、α、検出力から必要なサンプルサイズを決定することです。\n";

    // ============================================================
    // 2. 1標本t検定のパワー解析
    // ============================================================
    print_section("2. 1標本t検定のパワー解析");

    print_subsection("2.1 基本的な検出力の計算");
    std::cout << "既知のサンプルサイズと効果量から検出力を計算します。\n\n";

    std::cout << "例: 新しい教育プログラムの効果を検証\n";
    std::cout << "    - 既存の標準スコア平均: μ₀ = 100\n";
    std::cout << "    - 期待される改善後の平均: μ₁ = 105\n";
    std::cout << "    - 標準偏差: σ = 15\n";
    std::cout << "    - 利用可能なサンプルサイズ: n = 30\n\n";

    double cohen_d_education = (105.0 - 100.0) / 15.0;
    std::cout << "効果量 (Cohen's d) = (105 - 100) / 15 = " << cohen_d_education << "\n";
    std::cout << "これは「小さい」効果量に分類されます。\n\n";

    double power_30 = statcpp::power_t_test_one_sample(cohen_d_education, 30);
    std::cout << "n=30 の場合の検出力: " << power_30 << "\n";
    std::cout << "→ 約 " << (power_30 * 100) << "% の確率で効果を検出できます\n\n";

    std::cout << "検出力が不十分な場合、より多くのサンプルが必要です：\n";
    double power_50 = statcpp::power_t_test_one_sample(cohen_d_education, 50);
    double power_100 = statcpp::power_t_test_one_sample(cohen_d_education, 100);
    std::cout << "n=50  の場合の検出力: " << power_50 << " (" << (power_50 * 100) << "%)\n";
    std::cout << "n=100 の場合の検出力: " << power_100 << " (" << (power_100 * 100) << "%)\n";

    print_subsection("2.2 必要なサンプルサイズの計算");
    std::cout << "目標とする検出力を達成するために必要なサンプルサイズを計算します。\n\n";

    std::cout << "例: 新薬の効果を検証する臨床試験の設計\n";
    std::cout << "    - 期待される効果量: d = 0.5（中程度）\n";
    std::cout << "    - 目標検出力: 80%\n";
    std::cout << "    - 有意水準: α = 0.05\n\n";

    std::size_t n_required = statcpp::sample_size_t_test_one_sample(0.5, 0.80, 0.05);
    std::cout << "必要なサンプルサイズ: n = " << n_required << "\n\n";

    std::cout << "異なる検出力レベルでの必要サンプルサイズ：\n";
    std::size_t n_70 = statcpp::sample_size_t_test_one_sample(0.5, 0.70);
    std::size_t n_80 = statcpp::sample_size_t_test_one_sample(0.5, 0.80);
    std::size_t n_90 = statcpp::sample_size_t_test_one_sample(0.5, 0.90);
    std::cout << "検出力 70%: n = " << n_70 << "\n";
    std::cout << "検出力 80%: n = " << n_80 << "\n";
    std::cout << "検出力 90%: n = " << n_90 << "\n";
    std::cout << "→ 検出力を上げるには、より多くのサンプルが必要です\n";

    print_subsection("2.3 片側検定と両側検定の比較");
    std::cout << "片側検定は、効果の方向が事前に分かっている場合に使用します。\n";
    std::cout << "同じ検出力を得るために必要なサンプルサイズが少なくなります。\n\n";

    std::cout << "効果量 d = 0.5、検出力 80% の場合：\n";
    std::size_t n_two = statcpp::sample_size_t_test_one_sample(0.5, 0.80, 0.05, "two.sided");
    std::size_t n_greater = statcpp::sample_size_t_test_one_sample(0.5, 0.80, 0.05, "greater");
    std::size_t n_less = statcpp::sample_size_t_test_one_sample(0.5, 0.80, 0.05, "less");

    std::cout << "両側検定: n = " << n_two << "\n";
    std::cout << "片側検定 (greater): n = " << n_greater << "\n";
    std::cout << "片側検定 (less): n = " << n_less << "\n\n";

    std::cout << "注意: 片側検定は事前に効果の方向を予測できる場合のみ使用してください。\n";
    std::cout << "      方向が不明な場合は、両側検定を使用するのが安全です。\n";

    // ============================================================
    // 3. 2標本t検定のパワー解析
    // ============================================================
    print_section("3. 2標本t検定のパワー解析");

    print_subsection("3.1 A/Bテストのサンプルサイズ設計");
    std::cout << "ウェブサイトの新デザイン（B）が既存デザイン（A）より優れているかを検証\n\n";

    std::cout << "背景情報：\n";
    std::cout << "  - 現在のコンバージョン率: 10%\n";
    std::cout << "  - 検出したい改善: 相対的に 20% の向上（10% → 12%）\n";
    std::cout << "  - ただし、連続値での効果量を想定: d = 0.3\n\n";

    std::size_t n_ab = statcpp::sample_size_t_test_two_sample(0.3, 0.80, 0.05);
    std::cout << "各グループに必要なサンプルサイズ: n = " << n_ab << "\n";
    std::cout << "総サンプルサイズ: " << (n_ab * 2) << " 人\n\n";

    std::cout << "実際のテストでの検出力の確認：\n";
    double power_ab = statcpp::power_t_test_two_sample(0.3, n_ab, n_ab);
    std::cout << "n1 = n2 = " << n_ab << " の場合の検出力: " << power_ab << "\n";

    print_subsection("3.2 不均等なサンプルサイズの場合");
    std::cout << "実験群と対照群のサンプルサイズが異なる場合があります。\n";
    std::cout << "例: 対照群のデータは豊富だが、実験群は高コスト\n\n";

    std::cout << "比率 1:2 でサンプルを配分する場合（実験群:対照群 = 1:2）：\n";
    std::size_t n_unequal = statcpp::sample_size_t_test_two_sample(0.5, 0.80, 0.05, 2.0);
    std::cout << "実験群のサンプルサイズ: n1 = " << n_unequal << "\n";
    std::cout << "対照群のサンプルサイズ: n2 = " << (n_unequal * 2) << "\n\n";

    std::cout << "均等配分 (1:1) と不均等配分 (1:2) の比較：\n";
    std::size_t n_equal = statcpp::sample_size_t_test_two_sample(0.5, 0.80, 0.05, 1.0);
    std::cout << "均等配分 (1:1): 各群 n = " << n_equal << ", 総数 = " << (n_equal * 2) << "\n";
    std::cout << "不均等配分 (1:2): n1 = " << n_unequal << ", n2 = " << (n_unequal * 2)
              << ", 総数 = " << (n_unequal * 3) << "\n\n";
    std::cout << "→ 不均等配分は総サンプルサイズが増加するため、可能な限り均等配分が効率的です。\n";

    print_subsection("3.3 実際のデータから効果量を推定");
    std::cout << "既存のパイロットスタディのデータから効果量を推定し、\n";
    std::cout << "本研究のサンプルサイズを設計します。\n\n";

    std::vector<double> pilot_control = {23.1, 25.3, 22.8, 24.5, 26.1, 23.9, 25.7, 24.2};
    std::vector<double> pilot_treatment = {26.4, 28.2, 27.1, 29.3, 26.8, 28.9, 27.5, 28.0};

    std::cout << "パイロットスタディのデータ：\n";
    std::cout << "対照群 (n=8): ";
    for (double x : pilot_control) std::cout << x << " ";
    std::cout << "\n治療群 (n=8): ";
    for (double x : pilot_treatment) std::cout << x << " ";
    std::cout << "\n\n";

    double d_pilot = statcpp::cohens_d_two_sample(pilot_treatment.begin(), pilot_treatment.end(),
                                                  pilot_control.begin(), pilot_control.end());
    std::cout << "推定される効果量 (Cohen's d): " << d_pilot << "\n";
    std::cout << "これは「大きい」効果量（d > 0.8）に該当します。\n\n";

    std::cout << "本研究で必要なサンプルサイズ（検出力 80%）：\n";
    std::size_t n_main = statcpp::sample_size_t_test_two_sample(d_pilot, 0.80);
    std::cout << "各群のサンプルサイズ: n = " << n_main << "\n\n";

    std::cout << "注意: パイロットスタディからの効果量推定は不安定な場合があります。\n";
    std::cout << "      保守的な推定（効果量を小さめに設定）を行うことが推奨されます。\n";

    // ============================================================
    // 4. 比率検定のパワー解析
    // ============================================================
    print_section("4. 比率検定のパワー解析");

    print_subsection("4.1 臨床試験における治療成功率の比較");
    std::cout << "新しい治療法（実験群）と標準治療（対照群）の成功率を比較します。\n\n";

    std::cout << "研究計画：\n";
    std::cout << "  - 標準治療の成功率: p1 = 0.60 (60%)\n";
    std::cout << "  - 新治療の期待成功率: p2 = 0.75 (75%)\n";
    std::cout << "  - 検出したい差: 15 パーセントポイント\n";
    std::cout << "  - 目標検出力: 80%\n";
    std::cout << "  - 有意水準: α = 0.05\n\n";

    std::size_t n_clinical = statcpp::sample_size_prop_test(0.60, 0.75, 0.80, 0.05);
    std::cout << "各群に必要なサンプルサイズ: n = " << n_clinical << "\n";
    std::cout << "総サンプルサイズ: " << (n_clinical * 2) << " 人\n\n";

    std::cout << "実際の検出力の確認：\n";
    double power_clinical = statcpp::power_prop_test(0.60, 0.75, n_clinical);
    std::cout << "計算された検出力: " << power_clinical << " (" << (power_clinical * 100) << "%)\n";

    print_subsection("4.2 異なる効果サイズでのサンプルサイズ");
    std::cout << "基準となる成功率 p1 = 0.50 からの改善を検出する場合\n\n";

    std::cout << "検出したい改善の大きさとサンプルサイズ：\n";
    double base_p = 0.50;
    std::vector<double> improvements = {0.05, 0.10, 0.15, 0.20};

    std::cout << std::setw(20) << "改善幅"
              << std::setw(15) << "p2"
              << std::setw(20) << "必要サンプル(各群)" << "\n";
    std::cout << std::string(55, '-') << "\n";

    for (double imp : improvements) {
        double p2 = base_p + imp;
        std::size_t n = statcpp::sample_size_prop_test(base_p, p2, 0.80, 0.05);
        std::cout << std::setw(15) << (imp * 100) << "%"
                  << std::setw(15) << p2
                  << std::setw(20) << n << "\n";
    }

    std::cout << "\n→ 検出したい効果が小さいほど、より多くのサンプルが必要になります。\n";

    print_subsection("4.3 ベースライン率が極端な場合");
    std::cout << "ベースライン率（p1）が極端（0に近い、または1に近い）場合、\n";
    std::cout << "同じ絶対差でも必要なサンプルサイズが変わります。\n\n";

    std::cout << "10パーセントポイントの改善を検出する場合：\n\n";

    std::vector<std::pair<double, double>> scenarios = {
        {0.10, 0.20},  // 低ベースライン
        {0.30, 0.40},  // 中低ベースライン
        {0.50, 0.60},  // 中間ベースライン
        {0.70, 0.80},  // 中高ベースライン
        {0.85, 0.95}   // 高ベースライン
    };

    std::cout << std::setw(15) << "p1"
              << std::setw(15) << "p2"
              << std::setw(20) << "必要サンプル" << "\n";
    std::cout << std::string(50, '-') << "\n";

    for (const auto& scenario : scenarios) {
        std::size_t n = statcpp::sample_size_prop_test(scenario.first, scenario.second, 0.80, 0.05);
        std::cout << std::setw(15) << scenario.first
                  << std::setw(15) << scenario.second
                  << std::setw(20) << n << "\n";
    }

    std::cout << "\n→ ベースライン率が0.5付近の場合に最も多くのサンプルが必要です。\n";
    std::cout << "   極端な率（0に近い、1に近い）の場合は比較的少ないサンプルで済みます。\n";

    // ============================================================
    // 5. サンプルサイズ設計の実践例
    // ============================================================
    print_section("5. サンプルサイズ設計の実践例");

    print_subsection("5.1 心理学研究: 認知訓練プログラムの効果検証");
    std::cout << "研究目的: 認知訓練プログラムがワーキングメモリを改善するか検証\n\n";

    std::cout << "先行研究からの情報：\n";
    std::cout << "  - 類似の介入研究で報告された効果量: d = 0.40 ~ 0.60\n";
    std::cout << "  - 保守的に d = 0.45 を想定\n\n";

    std::cout << "研究デザインの選択肢：\n\n";

    std::cout << "オプション1: 両側検定、検出力 80%\n";
    std::size_t n_psych_80 = statcpp::sample_size_t_test_two_sample(0.45, 0.80);
    std::cout << "  各群のサンプルサイズ: " << n_psych_80 << "\n";
    std::cout << "  総サンプルサイズ: " << (n_psych_80 * 2) << "\n\n";

    std::cout << "オプション2: 両側検定、検出力 90%（より保守的）\n";
    std::size_t n_psych_90 = statcpp::sample_size_t_test_two_sample(0.45, 0.90);
    std::cout << "  各群のサンプルサイズ: " << n_psych_90 << "\n";
    std::cout << "  総サンプルサイズ: " << (n_psych_90 * 2) << "\n\n";

    std::cout << "オプション3: 片側検定（改善のみを検出）、検出力 80%\n";
    std::size_t n_psych_one = statcpp::sample_size_t_test_two_sample(0.45, 0.80, 0.05, 1.0, "greater");
    std::cout << "  各群のサンプルサイズ: " << n_psych_one << "\n";
    std::cout << "  総サンプルサイズ: " << (n_psych_one * 2) << "\n\n";

    std::cout << "推奨: オプション1（両側検定、80%）が標準的な選択です。\n";
    std::cout << "      リソースに余裕があればオプション2も検討に値します。\n";

    print_subsection("5.2 医学研究: 新薬の有効性検証");
    std::cout << "研究目的: 新薬が血圧を低下させるか検証（プラセボ対照試験）\n\n";

    std::cout << "臨床的に意味のある効果の設定：\n";
    std::cout << "  - 収縮期血圧の低下: 5 mmHg\n";
    std::cout << "  - 集団の標準偏差: 12 mmHg\n";
    std::cout << "  - 効果量: d = 5/12 = " << (5.0/12.0) << "\n\n";

    double d_medical = 5.0 / 12.0;
    std::size_t n_medical = statcpp::sample_size_t_test_two_sample(d_medical, 0.80, 0.05);
    std::cout << "必要なサンプルサイズ（各群）: " << n_medical << "\n";
    std::cout << "総サンプルサイズ: " << (n_medical * 2) << "\n\n";

    std::cout << "ドロップアウト率を考慮した調整：\n";
    double dropout_rate = 0.15;  // 15%のドロップアウトを想定
    std::size_t n_adjusted = static_cast<std::size_t>(std::ceil(n_medical / (1.0 - dropout_rate)));
    std::cout << "  ドロップアウト率: " << (dropout_rate * 100) << "%\n";
    std::cout << "  調整後のサンプルサイズ（各群）: " << n_adjusted << "\n";
    std::cout << "  調整後の総サンプルサイズ: " << (n_adjusted * 2) << "\n\n";

    std::cout << "重要: 臨床試験では、ドロップアウトや不遵守を考慮して\n";
    std::cout << "        計算されたサンプルサイズよりも多めに募集することが一般的です。\n";

    print_subsection("5.3 マーケティング研究: メールキャンペーンの効果測定");
    std::cout << "研究目的: 新しいメールデザインがクリック率を改善するか検証\n\n";

    std::cout << "現状の分析：\n";
    std::cout << "  - 現在のクリック率: 3%\n";
    std::cout << "  - 目標クリック率: 4%（相対的に33%の改善）\n";
    std::cout << "  - 有意水準: α = 0.05\n";
    std::cout << "  - 目標検出力: 80%\n\n";

    std::size_t n_marketing = statcpp::sample_size_prop_test(0.03, 0.04, 0.80, 0.05);
    std::cout << "各バージョンに必要な送信数: " << n_marketing << "\n";
    std::cout << "総送信数: " << (n_marketing * 2) << "\n\n";

    std::cout << "より現実的な目標での再計算：\n";
    std::cout << "  目標クリック率: 3.5%（相対的に17%の改善）\n";
    std::size_t n_realistic = statcpp::sample_size_prop_test(0.03, 0.035, 0.80, 0.05);
    std::cout << "  各バージョンに必要な送信数: " << n_realistic << "\n";
    std::cout << "  総送信数: " << (n_realistic * 2) << "\n\n";

    std::cout << "洞察: 低い基準率からの小さな改善を検出するには、\n";
    std::cout << "      非常に大きなサンプルサイズが必要になります。\n";

    // ============================================================
    // 6. 効果量とパワーの関係
    // ============================================================
    print_section("6. 効果量とパワーの関係");

    print_subsection("6.1 効果量の大きさとサンプルサイズ");
    std::cout << "Cohenの基準に基づく効果量の分類と必要なサンプルサイズ：\n\n";

    std::vector<std::pair<std::string, double>> effect_sizes = {
        {"小（Small）", 0.2},
        {"中（Medium）", 0.5},
        {"大（Large）", 0.8}
    };

    std::cout << std::setw(20) << "効果量"
              << std::setw(10) << "d"
              << std::setw(20) << "サンプル(n, 80%)"
              << std::setw(20) << "サンプル(n, 90%)" << "\n";
    std::cout << std::string(70, '-') << "\n";

    for (const auto& es : effect_sizes) {
        std::size_t n_80 = statcpp::sample_size_t_test_two_sample(es.second, 0.80);
        std::size_t n_90 = statcpp::sample_size_t_test_two_sample(es.second, 0.90);
        std::cout << std::setw(20) << es.first
                  << std::setw(10) << es.second
                  << std::setw(20) << n_80
                  << std::setw(20) << n_90 << "\n";
    }

    std::cout << "\n重要な洞察：\n";
    std::cout << "  - 効果量が半分になると、必要なサンプルサイズは約4倍になります\n";
    std::cout << "  - 小さい効果を検出するには、非常に大きなサンプルが必要です\n";
    std::cout << "  - 研究デザインの段階で、検出可能な最小効果量を慎重に設定する必要があります\n";

    print_subsection("6.2 固定サンプルサイズでの検出力曲線");
    std::cout << "サンプルサイズ n=50（各群）の場合、効果量による検出力の変化：\n\n";

    std::size_t fixed_n = 50;
    std::vector<double> d_values = {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

    std::cout << std::setw(15) << "効果量 (d)"
              << std::setw(15) << "検出力"
              << "  " << "視覚化" << "\n";
    std::cout << std::string(60, '-') << "\n";

    for (double d : d_values) {
        double power = statcpp::power_t_test_two_sample(d, fixed_n, fixed_n);
        int bars = static_cast<int>(power * 40);  // スケール調整
        std::cout << std::setw(15) << d
                  << std::setw(15) << power
                  << "  " << std::string(bars, '#') << "\n";
    }

    std::cout << "\n→ 効果量が大きくなるほど、検出力は急速に向上します。\n";
    std::cout << "   n=50では、d=0.5（中程度の効果）で約70%の検出力が得られます。\n";

    // ============================================================
    // 7. α、β、n、効果量のトレードオフ
    // ============================================================
    print_section("7. α、β、n、効果量のトレードオフ");

    print_subsection("7.1 有意水準（α）の影響");
    std::cout << "より厳しい有意水準（小さいα）を設定すると、より多くのサンプルが必要になります。\n\n";

    double effect = 0.5;
    double power_target = 0.80;
    std::vector<double> alpha_levels = {0.10, 0.05, 0.01, 0.001};

    std::cout << std::setw(15) << "有意水準 (α)"
              << std::setw(20) << "必要サンプル(各群)"
              << std::setw(15) << "増加率" << "\n";
    std::cout << std::string(50, '-') << "\n";

    std::size_t baseline_n = 0;
    for (size_t i = 0; i < alpha_levels.size(); ++i) {
        std::size_t n = statcpp::sample_size_t_test_two_sample(effect, power_target, alpha_levels[i]);
        if (i == 0) baseline_n = n;
        double increase = (static_cast<double>(n) / baseline_n - 1.0) * 100;

        std::cout << std::setw(15) << alpha_levels[i]
                  << std::setw(20) << n
                  << std::setw(14) << (i == 0 ? "-" : "+" + std::to_string(static_cast<int>(increase)) + "%") << "\n";
    }

    std::cout << "\n考慮事項：\n";
    std::cout << "  - 多重比較を行う場合は、より厳しいα（例: 0.01）の使用を検討\n";
    std::cout << "  - 探索的研究では α = 0.10 も許容される場合があります\n";
    std::cout << "  - ただし、α = 0.05 が最も一般的な標準です\n";

    print_subsection("7.2 検出力（1-β）の選択");
    std::cout << "検出力の目標値を変更した場合の影響を比較します。\n\n";

    std::vector<double> power_levels = {0.70, 0.75, 0.80, 0.85, 0.90, 0.95};

    std::cout << std::setw(12) << "検出力"
              << std::setw(10) << "β"
              << std::setw(20) << "必要サンプル(各群)"
              << std::setw(15) << "増加率" << "\n";
    std::cout << std::string(57, '-') << "\n";

    std::size_t n_baseline = 0;
    for (size_t i = 0; i < power_levels.size(); ++i) {
        std::size_t n = statcpp::sample_size_t_test_two_sample(0.5, power_levels[i]);
        if (i == 2) n_baseline = n;  // 80%をベースライン
        double increase = (static_cast<double>(n) / n_baseline - 1.0) * 100;
        double beta = 1.0 - power_levels[i];

        std::cout << std::setw(12) << power_levels[i]
                  << std::setw(10) << beta
                  << std::setw(20) << n
                  << std::setw(14) << (i == 2 ? "-" : (increase >= 0 ? "+" : "") + std::to_string(static_cast<int>(increase)) + "%") << "\n";
    }

    std::cout << "\nガイドライン：\n";
    std::cout << "  - 80% (β=0.20): 最も一般的な標準\n";
    std::cout << "  - 90% (β=0.10): 重要な意思決定を伴う研究で推奨\n";
    std::cout << "  - 95% (β=0.05): 非常に保守的、臨床試験などで使用\n";

    print_subsection("7.3 効果量の不確実性への対処");
    std::cout << "効果量の推定には不確実性が伴います。\n";
    std::cout << "保守的なアプローチとして、期待値より小さい効果量を想定します。\n\n";

    std::cout << "例: 先行研究で d = 0.6 の効果が報告されている場合\n\n";

    std::vector<std::pair<std::string, double>> scenarios_es = {
        {"楽観的（報告値）", 0.60},
        {"現実的（80%）", 0.48},
        {"保守的（70%）", 0.42},
        {"非常に保守的（60%）", 0.36}
    };

    std::cout << std::setw(25) << "シナリオ"
              << std::setw(15) << "想定効果量"
              << std::setw(20) << "必要サンプル" << "\n";
    std::cout << std::string(60, '-') << "\n";

    for (const auto& scenario : scenarios_es) {
        std::size_t n = statcpp::sample_size_t_test_two_sample(scenario.second, 0.80);
        std::cout << std::setw(25) << scenario.first
                  << std::setw(15) << scenario.second
                  << std::setw(20) << n << "\n";
    }

    std::cout << "\n推奨アプローチ：\n";
    std::cout << "  1. 先行研究の効果量を過信しない（Publication biasの可能性）\n";
    std::cout << "  2. 複数の先行研究があれば、それらの平均または下限を使用\n";
    std::cout << "  3. パイロットスタディで効果量を推定する場合、信頼区間の下限を使用\n";
    std::cout << "  4. 不確実性が高い場合、保守的な（小さめの）効果量を想定\n";

    // ============================================================
    // 8. 実用的なガイドラインとベストプラクティス
    // ============================================================
    print_section("8. 実用的なガイドラインとベストプラクティス");

    print_subsection("8.1 研究デザインのチェックリスト");
    std::cout << "パワー解析を実施する際の標準的な手順：\n\n";

    std::cout << "□ ステップ1: 研究仮説を明確化\n";
    std::cout << "    - 帰無仮説と対立仮説を明示的に記述\n";
    std::cout << "    - 片側検定か両側検定かを決定\n\n";

    std::cout << "□ ステップ2: 効果量を設定\n";
    std::cout << "    - 先行研究からの情報を収集\n";
    std::cout << "    - 臨床的/実践的に意味のある最小効果量を特定\n";
    std::cout << "    - 保守的な推定を行う\n\n";

    std::cout << "□ ステップ3: 統計的パラメータを決定\n";
    std::cout << "    - 有意水準 α（通常 0.05）\n";
    std::cout << "    - 目標検出力（通常 0.80 以上）\n\n";

    std::cout << "□ ステップ4: サンプルサイズを計算\n";
    std::cout << "    - 適切な検定手法を選択\n";
    std::cout << "    - パワー解析を実行\n\n";

    std::cout << "□ ステップ5: 実現可能性を評価\n";
    std::cout << "    - リソース（予算、時間、人員）との照合\n";
    std::cout << "    - 必要に応じてパラメータを調整\n\n";

    std::cout << "□ ステップ6: ドロップアウトを考慮\n";
    std::cout << "    - 予想されるドロップアウト率で調整\n";
    std::cout << "    - 最終的な募集目標を設定\n\n";

    std::cout << "□ ステップ7: 文書化\n";
    std::cout << "    - すべての仮定と計算を研究計画書に記録\n";
    std::cout << "    - 事前登録（pre-registration）を検討\n";

    print_subsection("8.2 サンプルサイズが制約される場合の対応");
    std::cout << "理想的なサンプルサイズを確保できない場合の戦略：\n\n";

    std::cout << "戦略1: 達成可能な検出力を報告\n";
    std::size_t available_n = 30;
    std::vector<double> detectable_effects = {0.3, 0.5, 0.7, 0.9};

    std::cout << "  利用可能なサンプル: 各群 n = " << available_n << "\n";
    std::cout << "  検出可能な効果量（80%の検出力）：\n\n";

    for (double d : detectable_effects) {
        double achieved_power = statcpp::power_t_test_two_sample(d, available_n, available_n);
        std::cout << "    d = " << d << ": 検出力 = " << achieved_power
                  << " (" << (achieved_power * 100) << "%)"
                  << (achieved_power >= 0.80 ? " ✓" : "") << "\n";
    }

    std::cout << "\n戦略2: 効果量を大きくする工夫\n";
    std::cout << "  - より強力な介入を設計\n";
    std::cout << "  - 測定の精度を向上（分散を減らす）\n";
    std::cout << "  - より均質な参加者群を対象にする\n\n";

    std::cout << "戦略3: より効率的なデザインを採用\n";
    std::cout << "  - 対応のあるデザイン（被験者内計画）を検討\n";
    std::cout << "  - 共変量を用いた調整分析\n";
    std::cout << "  - 適応的デザインの採用\n\n";

    std::cout << "戦略4: 正直な報告\n";
    std::cout << "  - 達成された検出力を報告\n";
    std::cout << "  - 検出できない効果の範囲を明示\n";
    std::cout << "  - 結果の解釈に制限を明記\n";

    print_subsection("8.3 一般的な推奨値");
    std::cout << "様々な研究領域での標準的なパラメータ設定：\n\n";

    std::cout << "■ 基礎研究・探索的研究:\n";
    std::cout << "    α = 0.05 (両側), 検出力 = 0.80\n";
    std::cout << "    効果量: 中程度（d = 0.5）または先行研究に基づく\n\n";

    std::cout << "■ 臨床試験（確証的研究）:\n";
    std::cout << "    α = 0.05 (両側), 検出力 = 0.90\n";
    std::cout << "    効果量: 臨床的に意味のある最小差に基づく\n\n";

    std::cout << "■ 優越性試験（Superiority trial）:\n";
    std::cout << "    α = 0.05 (両側), 検出力 = 0.80-0.90\n";
    std::cout << "    効果量: 期待される効果に基づく\n\n";

    std::cout << "■ 非劣性試験（Non-inferiority trial）:\n";
    std::cout << "    α = 0.025 (片側), 検出力 = 0.80-0.90\n";
    std::cout << "    効果量: 非劣性マージンに基づく\n\n";

    std::cout << "■ パイロット研究:\n";
    std::cout << "    α = 0.10, 検出力 = 0.70-0.80\n";
    std::cout << "    効果量: 大きい効果（d = 0.8）を検出できる程度\n";

    // ============================================================
    // 9. よくある間違いと正しい解釈
    // ============================================================
    print_section("9. よくある間違いと正しい解釈");

    print_subsection("9.1 間違い: 研究終了後のパワー解析（Post-hoc power）");
    std::cout << "✗ 間違ったアプローチ:\n";
    std::cout << "   「研究を実施した後、有意でなかった結果に対して、\n";
    std::cout << "    観測された効果量を用いて検出力を計算する」\n\n";

    std::cout << "問題点:\n";
    std::cout << "  - Post-hoc powerは有意性検定のp値と完全に相関する\n";
    std::cout << "  - 追加情報を提供しない\n";
    std::cout << "  - 検出力の本来の目的（研究計画）とは異なる\n\n";

    std::cout << "✓ 正しいアプローチ:\n";
    std::cout << "   信頼区間を報告し、効果量の推定値と不確実性を示す\n";
    std::cout << "   例: 「平均差 = 2.3, 95% CI [-0.5, 5.1]」\n";

    print_subsection("9.2 間違い: 検出力を誤解する");
    std::cout << "✗ よくある誤解:\n";
    std::cout << "   「検出力 80% は、効果が存在する確率が 80% という意味だ」\n\n";

    std::cout << "✓ 正しい理解:\n";
    std::cout << "   検出力 80% の意味:\n";
    std::cout << "   「もし真の効果が存在する場合、それを統計的に有意と検出できる確率が 80%」\n\n";

    std::cout << "重要な区別:\n";
    std::cout << "  - 検出力は、真の効果が存在すると仮定した条件付き確率\n";
    std::cout << "  - 効果の存在確率とは異なる概念\n";
    std::cout << "  - ベイズ統計では事後確率として推定可能\n";

    print_subsection("9.3 間違い: 効果量の推定を過信する");
    std::cout << "✗ よくある間違い:\n";
    std::cout << "   「パイロットスタディで d = 0.8 だったので、本研究でもその値を使う」\n\n";

    std::cout << "問題点の例証:\n";
    std::vector<double> pilot_effect_estimates = {0.8, 0.7, 0.9, 0.75, 0.85};
    std::cout << "  仮想的な5つの小規模研究の効果量推定値:\n";
    std::cout << "  ";
    for (double e : pilot_effect_estimates) std::cout << e << " ";
    double mean_effect = 0.0;
    for (double e : pilot_effect_estimates) mean_effect += e;
    mean_effect /= pilot_effect_estimates.size();
    std::cout << "\n  平均: " << mean_effect << "\n\n";

    std::cout << "  これらの推定値は、真の効果量（例: 0.5）の周りで変動します。\n";
    std::cout << "  小サンプルでは偶然大きな値が出やすい（Winner's curse）\n\n";

    std::cout << "✓ 正しいアプローチ:\n";
    std::cout << "  1. 複数の研究がある場合、メタアナリシスの結果を使用\n";
    std::cout << "  2. 信頼区間の下限を使用\n";
    std::cout << "  3. 保守的な推定（報告値の70-80%）を採用\n";
    std::cout << "  4. 臨床的/実践的に意味のある最小効果量を基準にする\n";

    print_subsection("9.4 間違い: サンプルサイズの線形思考");
    std::cout << "✗ よくある間違い:\n";
    std::cout << "   「効果量が半分なら、サンプルサイズも2倍にすればいい」\n\n";

    std::cout << "実際の関係:\n";
    double d_base = 0.4;
    std::size_t n_base = statcpp::sample_size_t_test_two_sample(d_base, 0.80);
    double d_half = d_base / 2.0;
    std::size_t n_half = statcpp::sample_size_t_test_two_sample(d_half, 0.80);

    std::cout << "  効果量 d = " << d_base << " の場合: n = " << n_base << "\n";
    std::cout << "  効果量 d = " << d_half << " の場合: n = " << n_half << "\n";
    std::cout << "  増加倍率: " << (static_cast<double>(n_half) / n_base) << " 倍\n\n";

    std::cout << "✓ 正しい理解:\n";
    std::cout << "   必要なサンプルサイズは効果量の2乗に反比例します。\n";
    std::cout << "   効果量が半分になると、サンプルサイズは約4倍必要です。\n";

    print_subsection("9.5 間違い: αとβを混同する");
    std::cout << "重要な区別:\n\n";

    std::cout << "α (第1種過誤率):\n";
    std::cout << "  - 効果が実際には存在しないのに、「存在する」と結論づける確率\n";
    std::cout << "  - 研究者が直接設定（通常 0.05）\n";
    std::cout << "  - p値と比較される閾値\n\n";

    std::cout << "β (第2種過誤率):\n";
    std::cout << "  - 効果が実際に存在するのに、「検出できない」確率\n";
    std::cout << "  - 検出力 = 1 - β\n";
    std::cout << "  - サンプルサイズ、効果量、αに依存\n\n";

    std::cout << "バランスの例:\n";
    std::cout << "  標準的な設定: α = 0.05, β = 0.20 (検出力 = 0.80)\n";
    std::cout << "  → 第1種過誤を第2種過誤より重視（1:4の比率）\n";

    // ============================================================
    // 10. まとめと実践への応用
    // ============================================================
    print_section("10. まとめと実践への応用");

    std::cout << "■ パワー解析の本質的な目的:\n\n";
    std::cout << "1. 研究資源の最適配分\n";
    std::cout << "   - 不十分なサンプルで意味のない結果を避ける\n";
    std::cout << "   - 過剰なサンプルで資源を無駄にしない\n\n";

    std::cout << "2. 透明性と再現性の向上\n";
    std::cout << "   - 研究計画の事前登録\n";
    std::cout << "   - 仮説とサンプルサイズの正当化\n\n";

    std::cout << "3. 倫理的研究実践\n";
    std::cout << "   - 参加者への負担を最小化\n";
    std::cout << "   - 科学的に価値のある研究を保証\n\n";

    std::cout << "■ 実装のための推奨ワークフロー:\n\n";

    std::cout << "Phase 1: 計画段階\n";
    std::cout << "  ┌─────────────────────────────────┐\n";
    std::cout << "  │ 1. 研究仮説の明確化            │\n";
    std::cout << "  │ 2. 効果量の推定（保守的に）    │\n";
    std::cout << "  │ 3. パワー解析の実施            │\n";
    std::cout << "  │ 4. 実現可能性の評価            │\n";
    std::cout << "  └─────────────────────────────────┘\n";
    std::cout << "           ↓\n";
    std::cout << "Phase 2: 文書化\n";
    std::cout << "  ┌─────────────────────────────────┐\n";
    std::cout << "  │ 5. 研究計画書への記載          │\n";
    std::cout << "  │ 6. 事前登録（推奨）            │\n";
    std::cout << "  └─────────────────────────────────┘\n";
    std::cout << "           ↓\n";
    std::cout << "Phase 3: 実施\n";
    std::cout << "  ┌─────────────────────────────────┐\n";
    std::cout << "  │ 7. 計画通りにデータ収集        │\n";
    std::cout << "  │ 8. 逸脱がある場合は記録        │\n";
    std::cout << "  └─────────────────────────────────┘\n";
    std::cout << "           ↓\n";
    std::cout << "Phase 4: 報告\n";
    std::cout << "  ┌─────────────────────────────────┐\n";
    std::cout << "  │ 9. 効果量と信頼区間を報告      │\n";
    std::cout << "  │ 10. 計画からの逸脱を説明       │\n";
    std::cout << "  └─────────────────────────────────┘\n\n";

    std::cout << "■ 最終チェックリスト:\n\n";
    std::cout << "研究開始前に確認すべき項目：\n";
    std::cout << "  ☑ 効果量の設定根拠が明確か\n";
    std::cout << "  ☑ サンプルサイズ計算が文書化されているか\n";
    std::cout << "  ☑ ドロップアウト率が考慮されているか\n";
    std::cout << "  ☑ 統計検定の選択が適切か\n";
    std::cout << "  ☑ 片側/両側検定の選択が正当化されるか\n";
    std::cout << "  ☑ 目標サンプルサイズが実現可能か\n";
    std::cout << "  ☑ 代替案（サンプルが不足する場合）があるか\n\n";

    std::cout << "■ さらなる学習のために:\n\n";
    std::cout << "推奨文献：\n";
    std::cout << "  - Cohen, J. (1988). Statistical Power Analysis for the\n";
    std::cout << "    Behavioral Sciences (2nd ed.)\n";
    std::cout << "  - Faul et al. (2007). G*Power 3: A flexible statistical\n";
    std::cout << "    power analysis program. Behavior Research Methods.\n\n";

    std::cout << "オンラインリソース：\n";
    std::cout << "  - G*Power (フリーソフトウェア)\n";
    std::cout << "  - R package: pwr, WebPower\n";
    std::cout << "  - Python: statsmodels.stats.power\n\n";

    std::cout << "=====================================\n";
    std::cout << "パワー解析の例示プログラムを終了します\n";
    std::cout << "=====================================\n";

    return 0;
}
