/**
 * @file example_basic_statistics.cpp
 * @brief statcpp::basic_statistics.hpp のサンプルコード
 *
 * このファイルでは、basic_statistics.hpp で提供される基本的な統計関数の
 * 使い方を実践的な例を通じて説明します。
 *
 * 【提供される関数】
 * - sum()            : 合計
 * - count()          : データ数
 * - mean()           : 算術平均
 * - median()         : 中央値（※要ソート済みデータ）
 * - mode()           : 最頻値（単一）
 * - modes()          : 最頻値（複数）
 * - geometric_mean() : 幾何平均
 * - harmonic_mean()  : 調和平均
 * - trimmed_mean()   : トリム平均（※要ソート済みデータ）
 *
 * 【コンパイル方法】
 * g++ -std=c++17 -I/path/to/statcpp/include example_basic_statistics.cpp -o example_basic_statistics
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <string>

// statcpp の基本統計ヘッダー
#include "statcpp/basic_statistics.hpp"

// ============================================================================
// 結果表示用のヘルパー関数
// ============================================================================

void print_section(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

void print_subsection(const std::string& title) {
    std::cout << "\n--- " << title << " ---\n";
}

// ============================================================================
// サンプルデータの表示
// ============================================================================

/**
 * @brief サンプルデータを表示する
 */
void example_sample_data() {
    print_section("サンプルデータ");

    // 基本的な数値データ（テストの点数を想定）
    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};

    std::cout << "テストの点数データ: ";
    for (double s : scores) std::cout << s << " ";
    std::cout << "\n";
}

// ============================================================================
// 1. sum() - 合計
// ============================================================================

/**
 * @brief sum() の使用例
 *
 * 【目的】
 * sum() は、データの総和を計算します。
 * すべての値を足し合わせた結果を返します。
 *
 * 【使用場面】
 * - 売上の合計を計算
 * - 総得点を求める
 * - 累積値の計算
 */
void example_sum() {
    print_section("1. sum() - 合計");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};
    double total = statcpp::sum(scores.begin(), scores.end());
    std::cout << "合計点: " << total << "\n";

    // 整数データでも使用可能（戻り値の型は要素の型に依存）
    std::vector<int> integers = {1, 2, 3, 4, 5};
    int int_sum = statcpp::sum(integers.begin(), integers.end());
    std::cout << "整数の合計 (1+2+3+4+5): " << int_sum << "\n";
}

// ============================================================================
// 2. count() - データ数
// ============================================================================

/**
 * @brief count() の使用例
 *
 * 【目的】
 * count() は、データの個数（サンプルサイズ）を返します。
 *
 * 【使用場面】
 * - サンプルサイズの確認
 * - 平均計算の分母として使用
 * - データの検証
 */
void example_count() {
    print_section("2. count() - データ数");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};
    std::size_t n = statcpp::count(scores.begin(), scores.end());
    std::cout << "データ数: " << n << "人\n";
}

// ============================================================================
// 3. mean() - 算術平均
// ============================================================================

/**
 * @brief mean() の使用例
 *
 * 【目的】
 * mean() は、データの算術平均（相加平均）を計算します。
 * 全データの合計をデータ数で割った値です。
 *
 * 【数式】
 * mean = (x₁ + x₂ + ... + xₙ) / n
 *
 * 【使用場面】
 * - クラスの平均点を求める
 * - 平均気温、平均収入などの計算
 * - 最も一般的な「中心」の指標
 *
 * 【注意点】
 * - 外れ値の影響を受けやすい
 * - 空のデータに対しては例外をスロー
 */
void example_mean() {
    print_section("3. mean() - 算術平均");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};
    double avg = statcpp::mean(scores.begin(), scores.end());
    std::cout << "平均点: " << avg << "点\n";

    // 外れ値の影響を確認
    print_subsection("外れ値の影響");
    std::vector<double> with_outlier = {85, 90, 78, 92, 88, 75, 95, 82, 88, 200}; // 200点は外れ値
    double avg_with_outlier = statcpp::mean(with_outlier.begin(), with_outlier.end());
    std::cout << "外れ値なしの平均: " << avg << "\n";
    std::cout << "外れ値ありの平均: " << avg_with_outlier << " (200点の影響で上昇)\n";
}

// ============================================================================
// 4. median() - 中央値
// ============================================================================

/**
 * @brief median() の使用例
 *
 * 【目的】
 * median() は、データを昇順に並べたときの中央の値を返します。
 * - データ数が奇数の場合：真ん中の値
 * - データ数が偶数の場合：真ん中2つの値の平均
 *
 * 【重要】
 * ※ 入力データは事前にソートされている必要があります！
 *
 * 【使用場面】
 * - 外れ値の影響を受けにくい中心を求めたい場合
 * - 所得の中央値（平均より実態を反映）
 * - 不動産価格の中央値
 *
 * 【特徴】
 * - 外れ値に対してロバスト（頑健）
 * - 順序尺度のデータにも適用可能
 */
void example_median() {
    print_section("4. median() - 中央値");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};

    // ソートしてから中央値を計算
    std::vector<double> sorted_scores = scores;
    std::sort(sorted_scores.begin(), sorted_scores.end());

    std::cout << "ソート済みデータ: ";
    for (double s : sorted_scores) std::cout << s << " ";
    std::cout << "\n";

    double med = statcpp::median(sorted_scores.begin(), sorted_scores.end());
    std::cout << "中央値: " << med << "点\n";

    // 外れ値があっても中央値は影響を受けにくい
    print_subsection("外れ値に対するロバスト性");
    std::vector<double> with_outlier = {85, 90, 78, 92, 88, 75, 95, 82, 88, 200};
    std::vector<double> sorted_with_outlier = with_outlier;
    std::sort(sorted_with_outlier.begin(), sorted_with_outlier.end());
    double med_with_outlier = statcpp::median(sorted_with_outlier.begin(), sorted_with_outlier.end());
    std::cout << "外れ値なしの中央値: " << med << "\n";
    std::cout << "外れ値ありの中央値: " << med_with_outlier << " (外れ値の影響は限定的)\n";

    // 偶数個のデータ
    print_subsection("偶数個のデータの場合");
    std::vector<double> even_data = {10, 20, 30, 40};
    double med_even = statcpp::median(even_data.begin(), even_data.end());
    std::cout << "データ: 10, 20, 30, 40\n";
    std::cout << "中央値: " << med_even << " (20と30の平均)\n";
}

// ============================================================================
// 5. mode() - 最頻値
// ============================================================================

/**
 * @brief mode() の使用例
 *
 * 【目的】
 * mode() は、データの中で最も頻繁に出現する値を返します。
 *
 * 【使用場面】
 * - 最も売れている商品サイズ
 * - 最も多い回答
 * - カテゴリカルデータの代表値
 *
 * 【注意点】
 * - 複数の最頻値がある場合、最小の値を返します（決定論的動作）
 * - 複数の最頻値すべてが必要な場合は modes() を使用
 */
void example_mode() {
    print_section("5. mode() - 最頻値");

    // 88と90が2回ずつ出現するデータ
    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};

    std::cout << "元データ: ";
    for (double s : scores) std::cout << s << " ";
    std::cout << "\n";

    double mode_val = statcpp::mode(scores.begin(), scores.end());
    std::cout << "最頻値: " << mode_val << " (複数ある場合は最小値を返す)\n";

    // カテゴリカルデータの例
    print_subsection("カテゴリカルデータでの使用");
    std::vector<std::string> sizes = {"M", "L", "M", "S", "M", "L", "M", "XL"};
    std::string most_popular = statcpp::mode(sizes.begin(), sizes.end());
    std::cout << "サイズデータ: M, L, M, S, M, L, M, XL\n";
    std::cout << "最も多いサイズ: " << most_popular << "\n";
}

// ============================================================================
// 6. modes() - 複数の最頻値
// ============================================================================

/**
 * @brief modes() の使用例
 *
 * 【目的】
 * modes() は、最頻値をすべて返します（昇順でソートされたvector）。
 *
 * 【使用場面】
 * - 二峰性分布（bimodal distribution）の検出
 * - 同率1位の商品をすべて知りたい場合
 * - 複数回答可のアンケート分析
 */
void example_modes() {
    print_section("6. modes() - 複数の最頻値");

    // 88と90が同じ頻度（2回）で出現
    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};

    std::vector<double> all_modes = statcpp::modes(scores.begin(), scores.end());
    std::cout << "すべての最頻値: ";
    for (double m : all_modes) std::cout << m << " ";
    std::cout << "\n";
    std::cout << "(88と90が各2回出現で同率1位)\n";
}

// ============================================================================
// 7. geometric_mean() - 幾何平均
// ============================================================================

/**
 * @brief geometric_mean() の使用例
 *
 * 【目的】
 * geometric_mean() は、データの幾何平均を計算します。
 * n個の値の積のn乗根です。
 *
 * 【数式】
 * geometric_mean = (x₁ × x₂ × ... × xₙ)^(1/n)
 * = exp((log(x₁) + log(x₂) + ... + log(xₙ)) / n)
 *
 * 【使用場面】
 * - 成長率の平均（年平均成長率など）
 * - 比率やパーセンテージの平均
 * - 対数正規分布に従うデータの中心
 *
 * 【注意点】
 * - すべての値が正でなければなりません（0や負の値は不可）
 * - 算術平均以下の値になります（AM-GM不等式）
 */
void example_geometric_mean() {
    print_section("7. geometric_mean() - 幾何平均");

    // 年間成長率の例（1.05 = 5%成長、0.95 = 5%減少）
    std::vector<double> growth_rates = {1.05, 1.10, 0.95, 1.08, 1.03};
    std::cout << "年間成長率: 1.05, 1.10, 0.95, 1.08, 1.03\n";

    double arith_mean = statcpp::mean(growth_rates.begin(), growth_rates.end());
    double geom_mean = statcpp::geometric_mean(growth_rates.begin(), growth_rates.end());

    std::cout << "算術平均: " << arith_mean << "\n";
    std::cout << "幾何平均: " << geom_mean << "\n";
    std::cout << "→ 成長率の平均には幾何平均が適切\n";
    std::cout << "  5年後の累積: " << std::pow(geom_mean, 5)
              << " (初期値1.0からの倍率)\n";
}

// ============================================================================
// 8. harmonic_mean() - 調和平均
// ============================================================================

/**
 * @brief harmonic_mean() の使用例
 *
 * 【目的】
 * harmonic_mean() は、データの調和平均を計算します。
 * 逆数の算術平均の逆数です。
 *
 * 【数式】
 * harmonic_mean = n / (1/x₁ + 1/x₂ + ... + 1/xₙ)
 *
 * 【使用場面】
 * - 速度の平均（往復の平均速度など）
 * - 比率の平均
 * - P/E比率（株価収益率）の平均
 *
 * 【注意点】
 * - 0を含むデータには使用不可
 * - 算術平均、幾何平均より常に小さい値になります
 */
void example_harmonic_mean() {
    print_section("8. harmonic_mean() - 調和平均");

    // 往復の平均速度の例
    // 行き: 60 km/h、帰り: 40 km/h で同じ距離を移動
    std::vector<double> speeds = {60.0, 40.0};

    double arith_speed = statcpp::mean(speeds.begin(), speeds.end());
    double harm_speed = statcpp::harmonic_mean(speeds.begin(), speeds.end());

    std::cout << "行きの速度: 60 km/h\n";
    std::cout << "帰りの速度: 40 km/h\n";
    std::cout << "算術平均: " << arith_speed << " km/h (誤り)\n";
    std::cout << "調和平均: " << harm_speed << " km/h (正しい平均速度)\n";
    std::cout << "\n【解説】\n";
    std::cout << "距離をdとすると、\n";
    std::cout << "  行き: d/60 時間、帰り: d/40 時間\n";
    std::cout << "  合計距離: 2d、合計時間: d/60 + d/40 = d(2+3)/120 = 5d/120\n";
    std::cout << "  平均速度: 2d / (5d/120) = 240/5 = 48 km/h\n";

    // 3つの平均の大小関係を確認
    print_subsection("3つの平均の大小関係 (AM >= GM >= HM)");
    std::vector<double> positive_data = {2.0, 8.0};
    double am = statcpp::mean(positive_data.begin(), positive_data.end());
    double gm = statcpp::geometric_mean(positive_data.begin(), positive_data.end());
    double hm = statcpp::harmonic_mean(positive_data.begin(), positive_data.end());
    std::cout << "データ: 2, 8\n";
    std::cout << "算術平均 (AM): " << am << "\n";
    std::cout << "幾何平均 (GM): " << gm << "\n";
    std::cout << "調和平均 (HM): " << hm << "\n";
    std::cout << "常に AM >= GM >= HM が成立\n";
}

// ============================================================================
// 9. trimmed_mean() - トリム平均
// ============================================================================

/**
 * @brief trimmed_mean() の使用例
 *
 * 【目的】
 * trimmed_mean() は、データの両端から一定割合を除いた後の平均を計算します。
 * 外れ値の影響を軽減しつつ、中央値より多くの情報を使います。
 *
 * 【重要】
 * ※ 入力データは事前にソートされている必要があります！
 *
 * 【引数】
 * proportion: 片側の除外割合（0.0 ～ 0.5未満）
 *   - 0.1 → 下位10%と上位10%を除外
 *   - 0.25 → 下位25%と上位25%を除外（四分位平均）
 *
 * 【使用場面】
 * - 外れ値の影響を軽減したい場合
 * - オリンピックの採点（最高点と最低点を除外）
 * - 信頼性の高い中心の推定
 */
void example_trimmed_mean() {
    print_section("9. trimmed_mean() - トリム平均");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};

    // ソート済みデータを使用
    std::vector<double> sorted_scores = scores;
    std::sort(sorted_scores.begin(), sorted_scores.end());

    std::cout << "ソート済みデータ: ";
    for (double s : sorted_scores) std::cout << s << " ";
    std::cout << "\n\n";

    double mean_normal = statcpp::mean(sorted_scores.begin(), sorted_scores.end());
    double trimmed_10 = statcpp::trimmed_mean(sorted_scores.begin(), sorted_scores.end(), 0.1);
    double trimmed_20 = statcpp::trimmed_mean(sorted_scores.begin(), sorted_scores.end(), 0.2);

    std::cout << "通常の平均:           " << mean_normal << "\n";
    std::cout << "10%トリム平均 (両端1つずつ除外): " << trimmed_10 << "\n";
    std::cout << "20%トリム平均 (両端2つずつ除外): " << trimmed_20 << "\n";

    // 外れ値がある場合の効果
    print_subsection("外れ値がある場合のトリム平均の効果");
    std::vector<double> with_outlier = {85, 90, 78, 92, 88, 75, 95, 82, 88, 200};
    std::vector<double> sorted_with_outlier = with_outlier;
    std::sort(sorted_with_outlier.begin(), sorted_with_outlier.end());

    std::cout << "外れ値ありデータ (ソート済み): ";
    for (double s : sorted_with_outlier) std::cout << s << " ";
    std::cout << "\n";

    double mean_outlier = statcpp::mean(sorted_with_outlier.begin(), sorted_with_outlier.end());
    double trimmed_outlier = statcpp::trimmed_mean(sorted_with_outlier.begin(), sorted_with_outlier.end(), 0.1);

    std::cout << "通常の平均: " << mean_outlier << " (外れ値200の影響大)\n";
    std::cout << "10%トリム平均: " << trimmed_outlier << " (外れ値を除外)\n";
}

// ============================================================================
// 10. ラムダ式（射影）を使った高度な使用例
// ============================================================================

/**
 * @brief ラムダ式（射影）を使った高度な使用例
 *
 * 【目的】
 * 多くの関数には、ラムダ式（射影関数）を受け取るオーバーロードがあります。
 * これにより、構造体のメンバーや変換後の値に対して統計計算ができます。
 *
 * 【使用場面】
 * - 構造体のベクターから特定のフィールドの統計
 * - データの変換（対数変換など）後の統計
 * - 複雑なデータ構造の処理
 */
void example_projection() {
    print_section("10. ラムダ式（射影）を使った高度な使用例");

    // 構造体の例
    struct Student {
        std::string name;
        int math_score;
        int english_score;
    };

    std::vector<Student> students = {
        {"Alice", 85, 90},
        {"Bob", 78, 82},
        {"Charlie", 92, 88},
        {"Diana", 88, 95},
        {"Eve", 75, 80}
    };

    // 数学の点数の平均
    double math_avg = statcpp::mean(students.begin(), students.end(),
                                    [](const Student& s) { return s.math_score; });

    // 英語の点数の平均
    double english_avg = statcpp::mean(students.begin(), students.end(),
                                       [](const Student& s) { return s.english_score; });

    // 合計点の平均
    double total_avg = statcpp::mean(students.begin(), students.end(),
                                     [](const Student& s) { return s.math_score + s.english_score; });

    std::cout << "学生データ:\n";
    for (const auto& s : students) {
        std::cout << "  " << s.name << ": 数学=" << s.math_score
                  << ", 英語=" << s.english_score << "\n";
    }
    std::cout << "\n";
    std::cout << "数学の平均: " << math_avg << "点\n";
    std::cout << "英語の平均: " << english_avg << "点\n";
    std::cout << "合計点の平均: " << total_avg << "点\n";
}

// ============================================================================
// まとめ
// ============================================================================

/**
 * @brief まとめを表示する
 */
void print_summary() {
    print_section("まとめ：どの平均を使うべきか？");

    std::cout << R"(
┌─────────────────┬────────────────────────────────────────┐
│ 関数            │ 使用場面                               │
├─────────────────┼────────────────────────────────────────┤
│ mean()          │ 一般的な平均。外れ値がない場合         │
│ median()        │ 外れ値がある場合。所得、不動産価格など │
│ mode()          │ カテゴリデータ。最も人気のある選択肢   │
│ geometric_mean()│ 成長率、比率の平均                     │
│ harmonic_mean() │ 速度の平均、P/E比率の平均              │
│ trimmed_mean()  │ 外れ値を部分的に除外したい場合         │
└─────────────────┴────────────────────────────────────────┘

【注意事項】
- median() と trimmed_mean() は事前にソートが必要
- geometric_mean() は正の値のみ
- harmonic_mean() はゼロを含まない
- 空のデータに対しては例外がスローされる
)";
}

// ============================================================================
// メイン関数
// ============================================================================

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // 各サンプルを実行
    example_sample_data();
    example_sum();
    example_count();
    example_mean();
    example_median();
    example_mode();
    example_modes();
    example_geometric_mean();
    example_harmonic_mean();
    example_trimmed_mean();
    example_projection();

    // まとめを表示
    print_summary();

    return 0;
}
