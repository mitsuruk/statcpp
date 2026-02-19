/**
 * @file example_effect_size.cpp
 * @brief 効果量のサンプルコード
 *
 * Cohen's d、Hedges' g、Glass's delta、Eta-squared、
 * Omega-squared、相関係数の変換等の効果量指標の使用例を示します。
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include "statcpp/effect_size.hpp"

int main() {
    std::cout << "=== 効果量（Effect Size）の例 ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    // ============================================================================
    // 1. Cohen's d（1標本）
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "1. Cohen's d（1標本）" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n【概念】" << std::endl;
    std::cout << "標本平均と理論平均の差を標準偏差で標準化した効果量" << std::endl;
    std::cout << "d = (X̄ - μ) / s" << std::endl;

    std::vector<double> sample1 = {85, 90, 88, 92, 87, 89, 91, 86, 88, 90};
    double mu0 = 80.0;  // 理論平均

    std::cout << "\n【実例: テストスコアの改善】" << std::endl;
    std::cout << "新しい学習プログラム後のテストスコア（n=10）" << std::endl;
    std::cout << "従来の平均スコア: " << mu0 << "点" << std::endl;

    // 標本標準偏差を使用
    double cohens_d_value = statcpp::cohens_d(sample1.begin(), sample1.end(), mu0);
    std::cout << "\n--- 効果量の計算 ---" << std::endl;
    std::cout << "標本平均: " << statcpp::mean(sample1.begin(), sample1.end()) << "点" << std::endl;
    std::cout << "理論平均 (H0): " << mu0 << "点" << std::endl;
    std::cout << "Cohen's d: " << cohens_d_value << std::endl;

    auto magnitude = statcpp::interpret_cohens_d(cohens_d_value);
    std::cout << "Effect size magnitude: ";
    switch (magnitude) {
        case statcpp::effect_size_magnitude::negligible: std::cout << "Negligible"; break;
        case statcpp::effect_size_magnitude::small: std::cout << "Small"; break;
        case statcpp::effect_size_magnitude::medium: std::cout << "Medium"; break;
        case statcpp::effect_size_magnitude::large: std::cout << "Large"; break;
    }
    std::cout << std::endl;

    std::cout << "\nInterpretation guidelines for Cohen's d:" << std::endl;
    std::cout << "  < 0.2: Negligible" << std::endl;
    std::cout << "  0.2 - 0.5: Small" << std::endl;
    std::cout << "  0.5 - 0.8: Medium" << std::endl;
    std::cout << "  ≥ 0.8: Large" << std::endl;

    // ============================================================================
    // 2. Cohen's d（2標本）
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "2. Cohen's d（2標本）" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n【概念】" << std::endl;
    std::cout << "2つのグループの平均の差を統合標準偏差で標準化した効果量" << std::endl;
    std::cout << "d = (X̄₁ - X̄₂) / s_pooled" << std::endl;

    std::vector<double> control_group = {75, 78, 76, 80, 77, 79, 78, 76};
    std::vector<double> treatment_group = {85, 88, 90, 87, 89, 86, 88, 91};

    std::cout << "\n【実例: 新薬の効果】" << std::endl;
    std::cout << "対照群（従来薬）と治療群（新薬）の効果を比較" << std::endl;

    double d_two = statcpp::cohens_d_two_sample(
        control_group.begin(), control_group.end(),
        treatment_group.begin(), treatment_group.end()
    );

    std::cout << "\n--- 効果量の計算 ---" << std::endl;
    std::cout << "対照群の平均: "
              << statcpp::mean(control_group.begin(), control_group.end()) << "点" << std::endl;
    std::cout << "治療群の平均: "
              << statcpp::mean(treatment_group.begin(), treatment_group.end()) << "点" << std::endl;
    std::cout << "Cohen's d (2標本): " << d_two << std::endl;
    std::cout << "  → 両群の差の大きさを標準化した値" << std::endl;

    std::cout << "\n効果量の評価: ";
    switch (statcpp::interpret_cohens_d(d_two)) {
        case statcpp::effect_size_magnitude::negligible: std::cout << "無視できる（Negligible）"; break;
        case statcpp::effect_size_magnitude::small: std::cout << "小（Small）"; break;
        case statcpp::effect_size_magnitude::medium: std::cout << "中（Medium）"; break;
        case statcpp::effect_size_magnitude::large: std::cout << "大（Large）"; break;
    }
    std::cout << std::endl;
    std::cout << "  → 新薬は従来薬より効果が大きい" << std::endl;

    // ============================================================================
    // 3. Hedges' g（バイアス補正されたCohen's d）
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "3. Hedges' g（バイアス補正されたCohen's d）" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n【概念】" << std::endl;
    std::cout << "小サンプルサイズでのバイアスを補正したCohen's d" << std::endl;
    std::cout << "サンプルサイズが小さいほど補正の効果が大きい" << std::endl;

    std::cout << "\n【実例: バイアス補正の比較】" << std::endl;

    double g_one = statcpp::hedges_g(sample1.begin(), sample1.end(), mu0);
    std::cout << "\n--- 1標本の場合 ---" << std::endl;
    std::cout << "Hedges' g: " << g_one << std::endl;

    double g_two = statcpp::hedges_g_two_sample(
        control_group.begin(), control_group.end(),
        treatment_group.begin(), treatment_group.end()
    );
    std::cout << "\n--- 2標本の場合 ---" << std::endl;
    std::cout << "Hedges' g: " << g_two << std::endl;

    std::cout << "\n--- Cohen's d と Hedges' g の比較 ---" << std::endl;
    std::cout << "Cohen's d:  " << d_two << std::endl;
    std::cout << "Hedges' g:  " << g_two << std::endl;
    std::cout << "差:         " << std::abs(d_two - g_two) << std::endl;
    std::cout << "  → Hedges' g は小サンプルのバイアスを補正" << std::endl;
    std::cout << "  → サンプルサイズが大きいと両者はほぼ等しくなる" << std::endl;

    // ============================================================================
    // 4. Glass's Delta
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "4. Glass's Delta（対照群の標準偏差を使用）" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n【概念】" << std::endl;
    std::cout << "対照群（コントロール群）の標準偏差のみを標準化に使用" << std::endl;
    std::cout << "処置が分散に影響を与える可能性がある場合に有用" << std::endl;

    double delta = statcpp::glass_delta(
        control_group.begin(), control_group.end(),
        treatment_group.begin(), treatment_group.end()
    );

    std::cout << "\n【実例: 処置の効果】" << std::endl;
    std::cout << "対照群の標準偏差を基準として効果を評価" << std::endl;

    std::cout << "\n--- Glass's Delta の計算 ---" << std::endl;
    std::cout << "Glass's Δ: " << delta << std::endl;
    std::cout << "  → 対照群の標準偏差で標準化" << std::endl;

    std::cout << "\n対照群の標準偏差: "
              << statcpp::sample_stddev(control_group.begin(), control_group.end()) << std::endl;
    std::cout << "治療群の標準偏差: "
              << statcpp::sample_stddev(treatment_group.begin(), treatment_group.end()) << std::endl;
    std::cout << "  → 両群の分散が異なる場合、Glass's Δ が適切" << std::endl;

    // ============================================================================
    // 5. 相関係数とCohen's dの変換
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "5. 相関係数とCohen's dの変換" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n【概念】" << std::endl;
    std::cout << "効果量の異なる指標間で相互変換が可能" << std::endl;
    std::cout << "メタ分析などで異なる尺度を統一する際に有用" << std::endl;

    double r_from_d = statcpp::d_to_r(d_two);
    double d_from_r = statcpp::r_to_d(r_from_d);

    std::cout << "\n【実例: Cohen's d ⇔ 相関係数 r】" << std::endl;
    std::cout << "\n--- 変換の往復 ---" << std::endl;
    std::cout << "元のCohen's d: " << d_two << std::endl;
    std::cout << "→ 相関係数 r に変換: " << r_from_d << std::endl;
    std::cout << "→ Cohen's d に戻す: " << d_from_r << std::endl;
    std::cout << "  → 往復変換で元の値に戻ることを確認" << std::endl;

    // t値から相関係数への変換例
    double t_value = 3.5;
    double df = 28.0;
    double r_from_t = statcpp::t_to_r(t_value, df);
    std::cout << "\n【実例: t統計量から相関係数への変換】" << std::endl;
    std::cout << "t値: " << t_value << std::endl;
    std::cout << "自由度: " << df << std::endl;
    std::cout << "相関係数 r: " << r_from_t << std::endl;
    std::cout << "  → t検定の結果を相関係数で表現" << std::endl;

    // ============================================================================
    // 6. Eta-squared と Partial Eta-squared
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "6. Eta-squared (η²) と Partial Eta-squared" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n【概念】" << std::endl;
    std::cout << "分散分析（ANOVA）における効果量の指標" << std::endl;
    std::cout << "η² = SS(効果) / SS(全体)" << std::endl;
    std::cout << "全体の分散のうち、要因で説明される割合を示す" << std::endl;

    // ANOVA の例（仮想データ）
    double ss_effect = 150.0;
    double ss_total = 500.0;
    double eta2 = statcpp::eta_squared(ss_effect, ss_total);

    std::cout << "\n【実例: ANOVA の効果量】" << std::endl;
    std::cout << "\n--- Eta-squared の計算 ---" << std::endl;
    std::cout << "効果の平方和 SS(効果): " << ss_effect << std::endl;
    std::cout << "全体の平方和 SS(全体): " << ss_total << std::endl;
    std::cout << "η²: " << eta2 << std::endl;
    std::cout << "  → 全分散の" << (eta2 * 100) << "%を要因が説明" << std::endl;

    // F値からPartial η²を計算
    double f_value = 8.5;
    double df1 = 2.0;
    double df2 = 27.0;
    double partial_eta2 = statcpp::partial_eta_squared(f_value, df1, df2);

    std::cout << "\n--- F値からPartial η²を計算 ---" << std::endl;
    std::cout << "F(" << df1 << ", " << df2 << ") = " << f_value << std::endl;
    std::cout << "Partial η²: " << partial_eta2 << std::endl;
    std::cout << "  → 他の要因を除いた場合の効果の大きさ" << std::endl;

    std::cout << "\n効果量の評価: ";
    switch (statcpp::interpret_eta_squared(partial_eta2)) {
        case statcpp::effect_size_magnitude::negligible: std::cout << "無視できる（Negligible）"; break;
        case statcpp::effect_size_magnitude::small: std::cout << "小（Small）"; break;
        case statcpp::effect_size_magnitude::medium: std::cout << "中（Medium）"; break;
        case statcpp::effect_size_magnitude::large: std::cout << "大（Large）"; break;
    }
    std::cout << std::endl;

    std::cout << "\n【η²の解釈ガイドライン】" << std::endl;
    std::cout << "  < 0.01: 無視できる（Negligible）" << std::endl;
    std::cout << "  0.01 - 0.06: 小（Small）" << std::endl;
    std::cout << "  0.06 - 0.14: 中（Medium）" << std::endl;
    std::cout << "  ≥ 0.14: 大（Large）" << std::endl;

    // ============================================================================
    // 7. Omega-squared（バイアス補正されたEta-squared）
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "7. Omega-squared (ω²)（バイアス補正版）" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n【概念】" << std::endl;
    std::cout << "小サンプルでのバイアスを補正したEta-squared" << std::endl;
    std::cout << "η²は効果を過大評価する傾向があり、ω²はそれを補正" << std::endl;

    double ms_error = 12.5;
    double df_effect = 2.0;
    double omega2 = statcpp::omega_squared(ss_effect, ss_total, ms_error, df_effect);

    std::cout << "\n【実例: バイアス補正の比較】" << std::endl;
    std::cout << "\n--- Omega-squared の計算 ---" << std::endl;
    std::cout << "効果の平方和 SS(効果): " << ss_effect << std::endl;
    std::cout << "全体の平方和 SS(全体): " << ss_total << std::endl;
    std::cout << "誤差の平均平方 MS(誤差): " << ms_error << std::endl;
    std::cout << "効果の自由度 df(効果): " << df_effect << std::endl;

    std::cout << "\n--- η² と ω² の比較 ---" << std::endl;
    std::cout << "η² (Eta-squared):   " << eta2 << std::endl;
    std::cout << "ω² (Omega-squared): " << omega2 << std::endl;
    std::cout << "差:                 " << (eta2 - omega2) << std::endl;
    std::cout << "\n  → ω²は小サンプルのバイアスを補正" << std::endl;
    std::cout << "  → 通常、ω² < η² となる（より保守的な推定）" << std::endl;

    // ============================================================================
    // 8. Cohen's h（比率の効果量）
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "8. Cohen's h（比率の効果量）" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n【概念】" << std::endl;
    std::cout << "2つの比率（割合）の差を測る効果量" << std::endl;
    std::cout << "逆正弦変換（arcsine transformation）を使用" << std::endl;

    double p1 = 0.65;  // グループ1の成功率
    double p2 = 0.45;  // グループ2の成功率

    double h = statcpp::cohens_h(p1, p2);

    std::cout << "\n【実例: コンバージョン率の比較】" << std::endl;
    std::cout << "A/Bテストで2つのランディングページを比較" << std::endl;

    std::cout << "\n--- Cohen's h の計算 ---" << std::endl;
    std::cout << "ページAのコンバージョン率: " << (p1 * 100) << "%" << std::endl;
    std::cout << "ページBのコンバージョン率: " << (p2 * 100) << "%" << std::endl;
    std::cout << "Cohen's h: " << h << std::endl;
    std::cout << "  → 比率の差の大きさを標準化した値" << std::endl;

    std::cout << "\n【Cohen's h の解釈ガイドライン】" << std::endl;
    std::cout << "  < 0.2: 小（Small）" << std::endl;
    std::cout << "  0.2 - 0.5: 中（Medium）" << std::endl;
    std::cout << "  ≥ 0.5: 大（Large）" << std::endl;

    // ============================================================================
    // 9. オッズ比と相対リスク
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "9. オッズ比（Odds Ratio）と相対リスク（Risk Ratio）" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n【概念】" << std::endl;
    std::cout << "2×2分割表における効果量の指標" << std::endl;
    std::cout << "オッズ比: (a/b) / (c/d) = ad/bc" << std::endl;
    std::cout << "相対リスク: [a/(a+b)] / [c/(c+d)]" << std::endl;

    // 2x2 分割表の例:
    //        Disease+  Disease-
    // Exposed    30        70      (a=30, b=70)
    // Control    10        90      (c=10, d=90)

    double a = 30, b = 70, c = 10, d = 90;

    std::cout << "\n【実例: 疫学研究】" << std::endl;
    std::cout << "喫煙と肺疾患の関連性を調査" << std::endl;

    std::cout << "\n--- 2×2分割表 ---" << std::endl;
    std::cout << "                疾患あり  疾患なし" << std::endl;
    std::cout << "  曝露あり        " << a << "        " << b << std::endl;
    std::cout << "  曝露なし        " << c << "        " << d << std::endl;

    double or_val = statcpp::odds_ratio(a, b, c, d);
    double rr_val = statcpp::risk_ratio(a, b, c, d);

    std::cout << "\n--- 効果量の計算 ---" << std::endl;
    std::cout << "オッズ比 (Odds Ratio): " << or_val << std::endl;
    std::cout << "  → 曝露群のオッズが非曝露群の" << or_val << "倍" << std::endl;

    std::cout << "\n相対リスク (Risk Ratio): " << rr_val << std::endl;
    std::cout << "  → 曝露群のリスクが非曝露群の" << rr_val << "倍" << std::endl;

    double risk_exposed = a / (a + b);
    double risk_control = c / (c + d);
    std::cout << "\n--- リスクの詳細 ---" << std::endl;
    std::cout << "曝露群のリスク: " << risk_exposed << " (" << (risk_exposed * 100) << "%)" << std::endl;
    std::cout << "非曝露群のリスク: " << risk_control << " (" << (risk_control * 100) << "%)" << std::endl;
    std::cout << "リスク差: " << (risk_exposed - risk_control) << std::endl;

    // ============================================================================
    // 10. 相関係数の解釈
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "10. 相関係数の解釈" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n【概念】" << std::endl;
    std::cout << "相関係数 r の絶対値によって関連性の強さを評価" << std::endl;
    std::cout << "r² は決定係数として解釈可能（説明される分散の割合）" << std::endl;

    std::vector<double> correlations = {0.05, 0.15, 0.35, 0.55, 0.85};

    std::cout << "\n【実例: 相関係数の強さ】" << std::endl;
    std::cout << "\n--- 様々な相関係数の解釈 ---" << std::endl;
    for (double r : correlations) {
        std::cout << "  r = " << std::setw(4) << r << ": ";
        switch (statcpp::interpret_correlation(r)) {
            case statcpp::effect_size_magnitude::negligible: std::cout << "無視できる（Negligible）"; break;
            case statcpp::effect_size_magnitude::small: std::cout << "弱い（Small）"; break;
            case statcpp::effect_size_magnitude::medium: std::cout << "中程度（Medium）"; break;
            case statcpp::effect_size_magnitude::large: std::cout << "強い（Large）"; break;
        }
        std::cout << " (r² = " << (r * r) << ", 説明される分散: " << (r * r * 100) << "%)" << std::endl;
    }

    std::cout << "\n【相関係数の解釈ガイドライン】" << std::endl;
    std::cout << "  < 0.1: 無視できる（Negligible）" << std::endl;
    std::cout << "  0.1 - 0.3: 弱い（Small）" << std::endl;
    std::cout << "  0.3 - 0.5: 中程度（Medium）" << std::endl;
    std::cout << "  ≥ 0.5: 強い（Large）" << std::endl;

    // ============================================================================
    // 11. 効果量の比較とまとめ
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "11. 効果量の比較とまとめ" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n【2標本比較の効果量まとめ】" << std::endl;
    std::cout << "\n--- 様々な効果量指標 ---" << std::endl;
    std::cout << "Cohen's d:       " << d_two << std::endl;
    std::cout << "  → 統合標準偏差で標準化" << std::endl;
    std::cout << "Hedges' g:       " << g_two << std::endl;
    std::cout << "  → 小サンプルのバイアス補正版" << std::endl;
    std::cout << "Glass's Δ:       " << delta << std::endl;
    std::cout << "  → 対照群の標準偏差で標準化" << std::endl;
    std::cout << "相関係数 r に変換:  " << r_from_d << std::endl;
    std::cout << "  → 関連性の強さとして表現" << std::endl;

    std::cout << "\n【総合評価】" << std::endl;
    std::cout << "すべての指標が示す効果の大きさ: ";
    switch (statcpp::interpret_cohens_d(d_two)) {
        case statcpp::effect_size_magnitude::negligible: std::cout << "無視できる（Negligible）"; break;
        case statcpp::effect_size_magnitude::small: std::cout << "小（Small）"; break;
        case statcpp::effect_size_magnitude::medium: std::cout << "中（Medium）"; break;
        case statcpp::effect_size_magnitude::large: std::cout << "大（Large）"; break;
    }
    std::cout << std::endl;

    std::cout << "\n【効果量使用のガイドライン】" << std::endl;
    std::cout << "1. 統計的有意性だけでなく、効果の大きさも報告する" << std::endl;
    std::cout << "2. 小サンプルの場合はバイアス補正版（Hedges' g, ω²）を使用" << std::endl;
    std::cout << "3. メタ分析では統一された効果量指標に変換する" << std::endl;
    std::cout << "4. 文脈に応じて適切な効果量指標を選択する" << std::endl;

    std::cout << "\n=== Example completed successfully ===" << std::endl;

    return 0;
}
