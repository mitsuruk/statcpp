/**
 * @file example_resampling.cpp
 * @brief リサンプリング手法のサンプルコード
 *
 * ブートストラップ法、ジャックナイフ法、置換検定、交差検証等の
 * リサンプリング手法の使用例を示します。
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include "statcpp/resampling.hpp"
#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"

int main()
{
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "=============================================================\n";
    std::cout << "          statcpp Resampling Methods Examples               \n";
    std::cout << "=============================================================\n\n";

    std::cout << "リサンプリング法とは？\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "リサンプリング法は、観測データから繰り返しサンプルを生成し\n";
    std::cout << "統計量の分布を推定する手法です。\n\n";
    std::cout << "主な利点：\n";
    std::cout << "  • 理論的な分布仮定が不要\n";
    std::cout << "  • 複雑な統計量にも適用可能\n";
    std::cout << "  • 有限サンプルでの推論が可能\n\n";

    std::cout << "=============================================================\n";
    std::cout << "1. Bootstrap法 - 基本概念\n";
    std::cout << "=============================================================\n\n";

    std::cout << "Bootstrap法の仕組み：\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "元のデータ: [12, 13, 11, 13, 14]  (n=5)\n\n";
    std::cout << "復元抽出で同じサイズのサンプルを繰り返し生成：\n";
    std::cout << "  Bootstrap Sample 1: [12, 12, 13, 11, 14]  → 平均: 12.4\n";
    std::cout << "  Bootstrap Sample 2: [13, 14, 11, 13, 13]  → 平均: 12.8\n";
    std::cout << "  Bootstrap Sample 3: [11, 12, 12, 13, 14]  → 平均: 12.4\n";
    std::cout << "  ...（繰り返し）\n";
    std::cout << "  Bootstrap Sample B: [13, 13, 14, 12, 13]  → 平均: 13.0\n\n";
    std::cout << "これらの統計量の分布から標準誤差や信頼区間を推定\n\n";

    // Sample data
    std::vector<double> data = {12.5, 13.2, 11.8, 12.9, 13.5, 12.1, 13.8, 12.7};

    std::cout << "=============================================================\n";
    std::cout << "2. Bootstrap法 - 平均の信頼区間\n";
    std::cout << "=============================================================\n\n";

    std::cout << "実データ例: 電池の寿命（時間）\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "測定値: ";
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << data[i];
        if (i < data.size() - 1) std::cout << ", ";
    }
    std::cout << "\n\n";

    statcpp::set_seed(123);  // For reproducibility

    auto boot_mean = statcpp::bootstrap_mean(
        data.begin(), data.end(),
        1000,  // number of bootstrap samples
        0.95   // confidence level
    );

    double original_mean = statcpp::mean(data.begin(), data.end());

    std::cout << "元データの平均:     " << original_mean << " 時間\n";
    std::cout << "Bootstrap推定値:    " << boot_mean.estimate << " 時間\n";
    std::cout << "Bootstrap標準誤差:  " << boot_mean.standard_error << " 時間\n";
    std::cout << "95%信頼区間:       [" << boot_mean.ci_lower << ", " << boot_mean.ci_upper << "] 時間\n\n";

    std::cout << "解釈：\n";
    std::cout << "  真の母集団平均は95%の確率で [" << boot_mean.ci_lower << ", "
              << boot_mean.ci_upper << "] の範囲にあると推定されます。\n\n";

    std::cout << "Bootstrap標準誤差の意味：\n";
    std::cout << "  サンプル平均のばらつきの尺度です。\n";
    std::cout << "  もし同じ実験を繰り返したら、平均は約±" << boot_mean.standard_error
              << "時間の範囲でばらつくと予想されます。\n\n";

    std::cout << "=============================================================\n";
    std::cout << "3. Bootstrap法 - 中央値の信頼区間\n";
    std::cout << "=============================================================\n\n";

    std::cout << "中央値（Median）は外れ値に頑健な統計量です。\n";
    std::cout << "通常の理論的方法では中央値の信頼区間を求めるのは困難ですが、\n";
    std::cout << "Bootstrap法なら簡単に推定できます。\n\n";

    std::cout << "外れ値を含むデータ例: 従業員の残業時間（月）\n";
    std::cout << "-------------------------------------------------------------\n";
    std::vector<double> overtime = {5.2, 6.1, 4.8, 5.9, 6.5, 5.3, 32.7, 6.2};
    std::cout << "残業時間: ";
    for (size_t i = 0; i < overtime.size(); ++i) {
        std::cout << overtime[i];
        if (i < overtime.size() - 1) std::cout << ", ";
    }
    std::cout << " 時間\n";
    std::cout << "  ※ 32.7時間は外れ値（特異なケース）\n\n";

    statcpp::set_seed(456);
    auto boot_median = statcpp::bootstrap_median(
        overtime.begin(), overtime.end(),
        1000,
        0.95
    );

    double original_median = statcpp::median(overtime.begin(), overtime.end());
    double overtime_mean = statcpp::mean(overtime.begin(), overtime.end());

    std::cout << "平均:              " << overtime_mean << " 時間（外れ値の影響大）\n";
    std::cout << "中央値:            " << original_median << " 時間（頑健）\n\n";
    std::cout << "Bootstrap中央値:   " << boot_median.estimate << " 時間\n";
    std::cout << "Bootstrap標準誤差: " << boot_median.standard_error << " 時間\n";
    std::cout << "95%信頼区間:      [" << boot_median.ci_lower << ", " << boot_median.ci_upper << "] 時間\n\n";

    std::cout << "解釈：\n";
    std::cout << "  外れ値（32.7時間）があっても、中央値は約" << original_median
              << "時間で安定しています。\n";
    std::cout << "  平均（" << overtime_mean << "時間）は外れ値に大きく影響されています。\n";
    std::cout << "  このような場合、中央値の方が代表値として適切です。\n\n";

    std::cout << "=============================================================\n";
    std::cout << "4. Bootstrap法 - 標準偏差の信頼区間\n";
    std::cout << "=============================================================\n\n";

    std::cout << "品質管理の例: 製品の重量のばらつき\n";
    std::cout << "-------------------------------------------------------------\n";
    std::vector<double> weights = {100.2, 99.8, 100.5, 99.9, 100.3, 100.1, 99.7, 100.4};
    std::cout << "製品重量: ";
    for (size_t i = 0; i < weights.size(); ++i) {
        std::cout << weights[i];
        if (i < weights.size() - 1) std::cout << ", ";
    }
    std::cout << " g\n\n";

    statcpp::set_seed(789);
    auto boot_sd = statcpp::bootstrap_stddev(
        weights.begin(), weights.end(),
        1000,
        0.95
    );

    double original_sd = statcpp::stddev(weights.begin(), weights.end());

    std::cout << "標本標準偏差:      " << original_sd << " g\n";
    std::cout << "Bootstrap推定値:   " << boot_sd.estimate << " g\n";
    std::cout << "Bootstrap標準誤差: " << boot_sd.standard_error << " g\n";
    std::cout << "95%信頼区間:      [" << boot_sd.ci_lower << ", " << boot_sd.ci_upper << "] g\n\n";

    std::cout << "品質管理への応用：\n";
    std::cout << "  許容範囲: ±0.5g (99.5g ~ 100.5g)\n";
    std::cout << "  現在のばらつき（σ）: " << boot_sd.estimate << " g\n";
    std::cout << "  もし 99.5% が許容範囲内にあるべきなら、必要なσは約0.15g\n";
    std::cout << "  → 現在のプロセスは許容範囲内に収まっています\n\n";

    std::cout << "=============================================================\n";
    std::cout << "5. 並び替え検定（Permutation Test）- 基本概念\n";
    std::cout << "=============================================================\n\n";

    std::cout << "並び替え検定とは？\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "帰無仮説「2群に差がない」が真なら、データのラベルを\n";
    std::cout << "入れ替えても統計量の分布は変わらないはずです。\n\n";
    std::cout << "例: 新薬の効果検証\n";
    std::cout << "  プラセボ群: [12, 13, 11, 13]  平均: 12.25\n";
    std::cout << "  新薬群:     [14, 15, 15, 14]  平均: 14.50\n";
    std::cout << "  観測差:     14.50 - 12.25 = 2.25\n\n";
    std::cout << "並び替え例:\n";
    std::cout << "  Perm 1: [12,13,15,14] vs [11,13,14,15]  差: 0.00\n";
    std::cout << "  Perm 2: [12,15,11,14] vs [13,13,14,15]  差: 0.75\n";
    std::cout << "  Perm 3: [14,13,15,13] vs [12,11,14,15]  差: -0.25\n";
    std::cout << "  ...\n\n";
    std::cout << "これらの並び替えで生成された差の分布と、観測差を比較し\n";
    std::cout << "p値を計算します。\n\n";

    std::cout << "=============================================================\n";
    std::cout << "6. 並び替え検定 - 2群の平均差の検定\n";
    std::cout << "=============================================================\n\n";

    std::cout << "A/Bテストの例: 新UIの効果検証\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "問題: 新しいUIデザインでユーザーの滞在時間が変わったか？\n\n";

    std::vector<double> old_ui = {12.5, 13.2, 11.8, 12.9, 13.1, 12.3};
    std::vector<double> new_ui = {14.1, 15.3, 14.8, 15.0, 14.5, 14.9};

    std::cout << "旧UI滞在時間（分）: ";
    for (size_t i = 0; i < old_ui.size(); ++i) {
        std::cout << old_ui[i];
        if (i < old_ui.size() - 1) std::cout << ", ";
    }
    std::cout << "\n";
    std::cout << "新UI滞在時間（分）: ";
    for (size_t i = 0; i < new_ui.size(); ++i) {
        std::cout << new_ui[i];
        if (i < new_ui.size() - 1) std::cout << ", ";
    }
    std::cout << "\n\n";

    double mean_old = statcpp::mean(old_ui.begin(), old_ui.end());
    double mean_new = statcpp::mean(new_ui.begin(), new_ui.end());
    double observed_diff = mean_new - mean_old;

    std::cout << "旧UIの平均:        " << mean_old << " 分\n";
    std::cout << "新UIの平均:        " << mean_new << " 分\n";
    std::cout << "観測された差:      " << observed_diff << " 分\n\n";

    statcpp::set_seed(321);
    auto perm_result = statcpp::permutation_test_two_sample(
        old_ui.begin(), old_ui.end(),
        new_ui.begin(), new_ui.end(),
        1000
    );

    std::cout << "並び替え検定の結果:\n";
    std::cout << "  p値: " << perm_result.p_value << "\n\n";

    std::cout << "解釈:\n";
    if (perm_result.p_value < 0.05) {
        std::cout << "  p値 < 0.05 → 統計的に有意\n";
        std::cout << "  新UIは旧UIと比べて滞在時間に有意な差があります。\n";
        std::cout << "  新UIの導入を推奨します。\n\n";
    } else {
        std::cout << "  p値 ≥ 0.05 → 統計的に有意でない\n";
        std::cout << "  差があるように見えますが、偶然の可能性を排除できません。\n";
        std::cout << "  より多くのデータを収集する必要があります。\n\n";
    }

    std::cout << "=============================================================\n";
    std::cout << "7. 並び替え検定 - より厳しい基準での検証\n";
    std::cout << "=============================================================\n\n";

    std::cout << "医薬品の臨床試験の例\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "医薬品では厳格な基準（α = 0.01）を使用することが多い\n\n";

    std::vector<double> placebo = {78.2, 79.5, 77.8, 78.9, 79.2, 78.5, 79.1};
    std::vector<double> drug = {82.1, 83.4, 81.9, 82.7, 83.0, 82.3, 82.8};

    std::cout << "プラセボ群の回復率(%): ";
    for (size_t i = 0; i < placebo.size(); ++i) {
        std::cout << placebo[i];
        if (i < placebo.size() - 1) std::cout << ", ";
    }
    std::cout << "\n";
    std::cout << "新薬群の回復率(%):     ";
    for (size_t i = 0; i < drug.size(); ++i) {
        std::cout << drug[i];
        if (i < drug.size() - 1) std::cout << ", ";
    }
    std::cout << "\n\n";

    double mean_placebo = statcpp::mean(placebo.begin(), placebo.end());
    double mean_drug = statcpp::mean(drug.begin(), drug.end());
    double drug_diff = mean_drug - mean_placebo;

    std::cout << "プラセボ群の平均:  " << mean_placebo << " %\n";
    std::cout << "新薬群の平均:      " << mean_drug << " %\n";
    std::cout << "観測された差:      " << drug_diff << " %\n\n";

    statcpp::set_seed(654);
    auto perm_drug = statcpp::permutation_test_two_sample(
        placebo.begin(), placebo.end(),
        drug.begin(), drug.end(),
        2000  // より多くの並び替えで精度向上
    );

    std::cout << "並び替え検定の結果（2000回の並び替え）:\n";
    std::cout << "  p値: " << perm_drug.p_value << "\n\n";

    std::cout << "解釈（有意水準 α = 0.01）:\n";
    if (perm_drug.p_value < 0.01) {
        std::cout << "  p値 < 0.01 → 非常に強い統計的証拠\n";
        std::cout << "  新薬の効果は統計的に極めて有意です。\n";
        std::cout << "  臨床試験の次の段階に進むことが推奨されます。\n\n";
    } else if (perm_drug.p_value < 0.05) {
        std::cout << "  0.01 ≤ p値 < 0.05\n";
        std::cout << "  通常の基準では有意ですが、医薬品の厳格な基準（α=0.01）\n";
        std::cout << "  には達していません。追加の臨床試験が必要です。\n\n";
    } else {
        std::cout << "  p値 ≥ 0.05 → 統計的に有意でない\n";
        std::cout << "  新薬の効果を確認できませんでした。\n\n";
    }

    std::cout << "=============================================================\n";
    std::cout << "8. BootstrapとPermutation Testの使い分け\n";
    std::cout << "=============================================================\n\n";

    std::cout << "Bootstrap法（信頼区間の推定）:\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "目的:  統計量（平均、中央値など）の不確実性を定量化\n";
    std::cout << "使用場面:\n";
    std::cout << "  • パラメータの信頼区間を求めたい\n";
    std::cout << "  • 「真の値はどの範囲にあるか？」を知りたい\n";
    std::cout << "  • 標準誤差を推定したい\n";
    std::cout << "例:\n";
    std::cout << "  「新製品の平均満足度は95%信頼区間で7.2〜8.5点」\n";
    std::cout << "  「中央年収は450万〜520万円の範囲と推定される」\n\n";

    std::cout << "並び替え検定（仮説検定）:\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "目的:  2群間に統計的に有意な差があるかを検定\n";
    std::cout << "使用場面:\n";
    std::cout << "  • 「AとBに差があるか？」を検証したい\n";
    std::cout << "  • 効果の有無を統計的に判断したい\n";
    std::cout << "  • p値を計算したい\n";
    std::cout << "例:\n";
    std::cout << "  「新薬は統計的に有意な効果がある（p<0.01）」\n";
    std::cout << "  「男女間で有意な差は見られない（p=0.34）」\n\n";

    std::cout << "=============================================================\n";
    std::cout << "9. リサンプリング法の長所と短所\n";
    std::cout << "=============================================================\n\n";

    std::cout << "長所:\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "✓ 分布仮定が不要\n";
    std::cout << "  → 正規分布などの仮定をしなくてよい\n\n";
    std::cout << "✓ 複雑な統計量にも適用可能\n";
    std::cout << "  → 中央値、四分位範囲、相関係数なども扱える\n\n";
    std::cout << "✓ 直感的に理解しやすい\n";
    std::cout << "  → 「データから繰り返しサンプルを取る」というシンプルな考え方\n\n";
    std::cout << "✓ 有限サンプルで推論可能\n";
    std::cout << "  → 理論的な漸近的性質に頼らない\n\n";

    std::cout << "短所:\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "✗ 計算コストが高い\n";
    std::cout << "  → 数千回のリサンプリングが必要（現代のコンピュータでは通常問題なし）\n\n";
    std::cout << "✗ 元データの質に依存\n";
    std::cout << "  → 元データが偏っていると結果も偏る\n\n";
    std::cout << "✗ 非常に小さなサンプルでは不安定\n";
    std::cout << "  → n < 10 程度では信頼性が低下する可能性\n\n";

    std::cout << "=============================================================\n";
    std::cout << "10. 実践的なガイドライン\n";
    std::cout << "=============================================================\n\n";

    std::cout << "Bootstrap回数の選び方:\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "  • 探索的分析:     B = 200 〜 500\n";
    std::cout << "  • 通常の信頼区間: B = 1000 〜 2000\n";
    std::cout << "  • 厳密な推定:     B = 5000 〜 10000\n";
    std::cout << "  • 論文・公式報告: B = 10000 以上\n\n";

    std::cout << "並び替え回数の選び方:\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "  • 探索的分析:     N = 500 〜 1000\n";
    std::cout << "  • 通常の検定:     N = 1000 〜 5000\n";
    std::cout << "  • 厳密なp値:      N = 10000 以上\n";
    std::cout << "  ※ 小さなp値（p<0.01）を正確に推定したい場合は多めに\n\n";

    std::cout << "信頼区間の水準の選び方:\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "  • 90% CI: 予備的な分析、スクリーニング\n";
    std::cout << "  • 95% CI: 標準的な科学研究（最も一般的）\n";
    std::cout << "  • 99% CI: 重要な意思決定、医療・安全関連\n\n";

    std::cout << "有意水準の選び方:\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "  • α = 0.10: 探索的研究\n";
    std::cout << "  • α = 0.05: 標準的な科学研究（最も一般的）\n";
    std::cout << "  • α = 0.01: 医薬品、重要な政策決定\n";
    std::cout << "  • α = 0.001: 極めて重要な意思決定\n\n";

    std::cout << "乱数シードの設定:\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "  statcpp::set_seed(123);  // 再現性のために必須\n";
    std::cout << "  論文やレポートでは必ずシード値を記載しましょう。\n\n";

    std::cout << "サンプルサイズのガイドライン:\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "  • n < 10:   リサンプリング法は不安定、注意が必要\n";
    std::cout << "  • n = 10-30: Bootstrap法は使えるが、解釈は慎重に\n";
    std::cout << "  • n > 30:   Bootstrap法は安定して動作\n";
    std::cout << "  • n > 100:  理論的手法とBootstrap法の結果がほぼ一致\n\n";

    std::cout << "=============================================================\n";
    std::cout << "11. よくある間違いと正しい解釈\n";
    std::cout << "=============================================================\n\n";

    std::cout << "❌ 間違い 1: 「95%信頼区間は真の値を95%の確率で含む」\n";
    std::cout << "✓  正しい:   「同じ手順を繰り返すと、95%の区間が真の値を含む」\n";
    std::cout << "   解説: 真の値は固定で、区間が変動する概念です。\n\n";

    std::cout << "❌ 間違い 2: 「p値は仮説が正しい確率」\n";
    std::cout << "✓  正しい:   「帰無仮説のもとで、観測結果以上の極端な\n";
    std::cout << "              結果が得られる確率」\n";
    std::cout << "   解説: p値は仮説の確率ではなく、データの確率です。\n\n";

    std::cout << "❌ 間違い 3: 「p > 0.05 なら差がない」\n";
    std::cout << "✓  正しい:   「p > 0.05 なら差があることを示す十分な証拠がない」\n";
    std::cout << "   解説: 「差がない」と「差を検出できない」は別物です。\n\n";

    std::cout << "❌ 間違い 4: 「統計的有意 = 実質的に重要」\n";
    std::cout << "✓  正しい:   「統計的有意性と実質的重要性は別」\n";
    std::cout << "   解説: n が大きいと小さな差でも有意になります。\n";
    std::cout << "          実務的に意味のある差かどうかは別途判断が必要です。\n\n";

    std::cout << "=============================================================\n";
    std::cout << "12. まとめ - いつどの手法を使うか\n";
    std::cout << "=============================================================\n\n";

    std::cout << "平均の信頼区間を求めたい:\n";
    std::cout << "  → bootstrap_mean() を使用\n";
    std::cout << "    例: 「新製品の平均満足度の95%信頼区間は？」\n\n";

    std::cout << "中央値の信頼区間を求めたい（外れ値に頑健）:\n";
    std::cout << "  → bootstrap_median() を使用\n";
    std::cout << "    例: 「年収の中央値の信頼区間は？」\n\n";

    std::cout << "ばらつき（標準偏差）の信頼区間を求めたい:\n";
    std::cout << "  → bootstrap_stddev() を使用\n";
    std::cout << "    例: 「製品の品質のばらつきはどの程度か？」\n\n";

    std::cout << "2群間に統計的に有意な差があるか検定したい:\n";
    std::cout << "  → permutation_test_two_sample() を使用\n";
    std::cout << "    例: 「新薬と従来薬に効果の差はあるか？」\n\n";

    std::cout << "分布の仮定をしたくない:\n";
    std::cout << "  → リサンプリング法（Bootstrap、並び替え検定）を使用\n";
    std::cout << "    正規分布などの仮定が不要です\n\n";

    std::cout << "複雑な統計量（中央値、四分位範囲など）を扱いたい:\n";
    std::cout << "  → Bootstrap法を使用\n";
    std::cout << "    理論的な手法が存在しない/複雑な場合に有効です\n\n";

    std::cout << "=============================================================\n";
    std::cout << "すべての例を実行しました！\n";
    std::cout << "=============================================================\n";

    return 0;
}
