/**
 * @file example_time_series.cpp
 * @brief 時系列解析のサンプルコード
 *
 * 自己相関関数(ACF)、偏自己相関関数(PACF)、移動平均、
 * 指数平滑化等の時系列解析手法の使用例を示します。
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>
#include "statcpp/time_series.hpp"

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

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // ============================================================================
    // 1. 自己相関関数 (ACF)
    // ============================================================================
    print_section("1. 自己相関関数 (ACF: Autocorrelation Function)");

    std::cout << R"(
【概念】
時系列データの値と、その過去の値との相関
ラグ（時間差）ごとの相関係数を計算

【実例: 季節パターンの検出】
周期的なパターンを持つデータで
どのラグで相関が高いかを確認
)";

    // シンプルな時系列データ（季節パターンを含む）
    std::vector<double> ts_data;
    for (int i = 0; i < 40; ++i) {
        double value = 10.0 + 5.0 * std::sin(2.0 * 3.14159 * i / 12.0) +
                       ((i % 3 == 0) ? 2.0 : -0.5);
        ts_data.push_back(value);
    }

    std::size_t max_lag = 10;
    auto acf_values = statcpp::acf(ts_data.begin(), ts_data.end(), max_lag);

    print_subsection("各ラグでの自己相関");
    std::cout << "  ラグ    ACF値\n";
    for (std::size_t lag = 0; lag <= max_lag; ++lag) {
        std::cout << std::setw(5) << lag << "  " << std::setw(8) << acf_values[lag];

        // 有意性の目安（±2/√n）
        double significance_level = 2.0 / std::sqrt(static_cast<double>(ts_data.size()));
        if (std::abs(acf_values[lag]) > significance_level && lag > 0) {
            std::cout << "  *";
        }
        std::cout << std::endl;
    }

    std::cout << "\n* は有意な自己相関を示す\n";
    std::cout << "有意性の境界: ±" << (2.0 / std::sqrt(static_cast<double>(ts_data.size()))) << "\n";
    std::cout << "→ この境界を超えるACF値は統計的に有意\n";

    // ============================================================================
    // 2. 偏自己相関関数 (PACF)
    // ============================================================================
    print_section("2. 偏自己相関関数 (PACF: Partial Autocorrelation Function)");

    std::cout << R"(
【概念】
中間ラグの影響を除いた、直接的な相関
ARモデルの次数決定に使用

【実例: ARモデルの次数決定】
どのラグまでが直接的な影響を持つかを特定
)";

    auto pacf_values = statcpp::pacf(ts_data.begin(), ts_data.end(), max_lag);

    print_subsection("各ラグでの偏自己相関");
    std::cout << "  ラグ   PACF値\n";
    for (std::size_t lag = 1; lag <= max_lag; ++lag) {
        std::cout << std::setw(5) << lag << "  " << std::setw(8) << pacf_values[lag - 1];

        double significance_level = 2.0 / std::sqrt(static_cast<double>(ts_data.size()));
        if (std::abs(pacf_values[lag - 1]) > significance_level) {
            std::cout << "  *";
        }
        std::cout << std::endl;
    }

    std::cout << "\n→ PACFはAR（自己回帰）モデルの次数特定に有用\n";
    std::cout << "→ 有意なPACFが途切れるラグがAR次数の候補\n";

    // ============================================================================
    // 3. 移動平均
    // ============================================================================
    print_section("3. 移動平均 (Moving Average)");

    std::cout << R"(
【概念】
一定期間の平均を計算してノイズを除去
トレンドを滑らかにして把握しやすくする

【実例: 売上データの平滑化】
日々の変動を除いて全体的なトレンドを確認
)";

    std::vector<double> sales_data = {100, 110, 105, 115, 120, 118, 125, 130, 128, 135};
    std::size_t window = 3;

    auto sma = statcpp::moving_average(sales_data.begin(), sales_data.end(), window);

    print_subsection(std::to_string(window) + "期間移動平均");
    std::cout << "  期間  売上   移動平均\n";

    for (std::size_t i = 0; i < sales_data.size(); ++i) {
        std::cout << std::setw(6) << (i + 1) << std::setw(6) << sales_data[i];
        if (i >= window - 1) {
            std::cout << std::setw(10) << sma[i - window + 1];
        } else {
            std::cout << "        --";
        }
        std::cout << std::endl;
    }
    std::cout << "→ 短期的な変動が平滑化され、トレンドが見やすくなる\n";

    // ============================================================================
    // 4. 指数移動平均
    // ============================================================================
    print_section("4. 指数移動平均 (EMA: Exponential Moving Average)");

    std::cout << R"(
【概念】
最近のデータに大きな重みを与える平均
単純移動平均より最新データに敏感

【実例: 最近のトレンド重視】
αパラメータで最新データへの反応度を調整
)";

    double alpha = 0.3;  // 平滑化パラメータ
    auto ema = statcpp::exponential_moving_average(sales_data.begin(), sales_data.end(), alpha);

    print_subsection("指数移動平均 (α = " + std::to_string(alpha) + ")");
    std::cout << "  期間  売上    EMA\n";

    for (std::size_t i = 0; i < sales_data.size(); ++i) {
        std::cout << std::setw(6) << (i + 1)
                  << std::setw(6) << sales_data[i]
                  << std::setw(8) << ema[i] << std::endl;
    }

    std::cout << "\n→ EMAは最近の観測値により大きな重みを与える\n";
    std::cout << "→ αが大きいほど変化に敏感に反応\n";

    // ============================================================================
    // 5. 差分系列
    // ============================================================================
    print_section("5. 差分系列 (Differencing for Stationarity)");

    std::cout << R"(
【概念】
現在の値と過去の値の差を計算
トレンドを除去して定常性を実現

【実例: トレンド除去】
増加傾向のあるデータから
トレンド成分を除去
)";

    // トレンドのあるデータ
    std::vector<double> trend_data = {100, 102, 105, 109, 114, 120, 127, 135, 144, 154};

    auto diff1 = statcpp::diff(trend_data.begin(), trend_data.end(), 1);

    print_subsection("元データと1階差分の比較");
    std::cout << "  期間  元データ  1階差分\n";

    for (std::size_t i = 0; i < trend_data.size(); ++i) {
        std::cout << std::setw(6) << (i + 1) << std::setw(10) << trend_data[i];
        if (i > 0) {
            std::cout << std::setw(10) << diff1[i - 1];
        } else {
            std::cout << "        --";
        }
        std::cout << std::endl;
    }

    std::cout << "\n→ 差分により上昇トレンドが除去される\n";
    std::cout << "→ 定常性の達成に必要な処理\n";

    // ============================================================================
    // 6. 季節差分
    // ============================================================================
    print_section("6. 季節差分 (Seasonal Differencing)");

    std::cout << R"(
【概念】
季節周期分だけ離れた値との差を計算
季節パターンを除去

【実例: 四半期データの季節性除去】
周期=4（四半期）のデータから
季節変動を取り除く
)";

    // 季節性のあるデータ（周期=4）
    std::vector<double> seasonal_data = {
        100, 80, 90, 110,  // Q1-Q4 Year 1
        105, 85, 95, 115,  // Q1-Q4 Year 2
        110, 90, 100, 120  // Q1-Q4 Year 3
    };

    std::size_t period = 4;
    auto seasonal_diff = statcpp::seasonal_diff(seasonal_data.begin(), seasonal_data.end(), period);

    print_subsection("季節データ（周期 = " + std::to_string(period) + "）");
    std::cout << "  四半期  値    季節差分\n";

    for (std::size_t i = 0; i < seasonal_data.size(); ++i) {
        std::cout << std::setw(8) << (i + 1) << std::setw(6) << seasonal_data[i];
        if (i >= period) {
            std::cout << std::setw(11) << seasonal_diff[i - period];
        } else {
            std::cout << "         --";
        }
        std::cout << std::endl;
    }

    std::cout << "\n→ 季節差分により季節パターンが除去される\n";
    std::cout << "→ 同じ季節（四半期）同士を比較\n";

    // ============================================================================
    // 7. ラグ系列
    // ============================================================================
    print_section("7. ラグ系列 (Lag Series)");

    std::cout << R"(
【概念】
データを時間軸上でずらした系列
過去の値を現在の値と並べて比較

【実例: 過去データとの比較】
2期前の価格と現在の価格を並べて分析
)";

    std::vector<double> price_data = {100, 102, 101, 103, 105, 104, 106};
    std::size_t lag = 2;

    auto lagged = statcpp::lag(price_data.begin(), price_data.end(), lag);

    print_subsection("ラグ" + std::to_string(lag) + "の系列");
    std::cout << "     t   価格  ラグ-" << lag << "\n";

    for (std::size_t i = 0; i < price_data.size(); ++i) {
        std::cout << std::setw(6) << (i + 1) << std::setw(7) << price_data[i];
        if (i >= lag) {
            std::cout << std::setw(8) << lagged[i - lag];
        } else {
            std::cout << "      --";
        }
        std::cout << std::endl;
    }
    std::cout << "→ ラグ系列は回帰分析の説明変数として使用可能\n";

    // ============================================================================
    // 8. 予測評価指標
    // ============================================================================
    print_section("8. 予測評価指標 (Forecast Evaluation Metrics)");

    std::cout << R"(
【概念】
予測精度を定量的に評価する指標
実測値と予測値の誤差を測定

【主な指標】
- MAE: 平均絶対誤差（外れ値に頑健）
- RMSE: 二乗平均平方根誤差（大きな誤差を重視）
- MAPE: 平均絶対パーセント誤差（相対誤差）
)";

    std::vector<double> actual = {100, 105, 110, 115, 120};
    std::vector<double> forecast = {98, 107, 108, 116, 122};

    // Calculate error metrics manually
    double mae = 0.0, mse = 0.0, mape = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        double error = actual[i] - forecast[i];
        mae += std::abs(error);
        mse += error * error;
        mape += std::abs(error / actual[i]) * 100.0;
    }
    mae /= actual.size();
    mse /= actual.size();
    double rmse = std::sqrt(mse);
    mape /= actual.size();

    print_subsection("予測値と実測値の比較");
    std::cout << "  期間  実測値  予測値   誤差\n";
    for (std::size_t i = 0; i < actual.size(); ++i) {
        std::cout << std::setw(6) << (i + 1)
                  << std::setw(8) << actual[i]
                  << std::setw(8) << forecast[i]
                  << std::setw(7) << (actual[i] - forecast[i]) << std::endl;
    }

    print_subsection("誤差指標");
    std::cout << "  MAE（平均絶対誤差）: " << mae << "\n";
    std::cout << "  MSE（平均二乗誤差）: " << mse << "\n";
    std::cout << "  RMSE（二乗平均平方根誤差）: " << rmse << "\n";
    std::cout << "  MAPE（平均絶対パーセント誤差）: " << mape << "%\n";
    std::cout << "\n→ 値が小さいほど予測精度が高い\n";

    // ============================================================================
    // 9. 実用例：売上予測
    // ============================================================================
    print_section("9. 実用例：売上データの分析");

    std::cout << R"(
【概念】
時系列分析手法を組み合わせて実データを分析
トレンド、季節性、自己相関を総合的に評価

【実例: 月次売上データ】
12ヶ月間の売上データから
トレンドとパターンを抽出
)";

    std::vector<double> monthly_sales = {
        120, 118, 125, 130, 128, 135, 140, 138, 145, 150, 148, 155
    };

    // 3ヶ月移動平均でトレンドを把握
    auto trend = statcpp::moving_average(monthly_sales.begin(), monthly_sales.end(), 3);

    print_subsection("月次売上データ（12ヶ月）とトレンド");
    std::cout << "  月  売上  3ヶ月MA トレンド\n";
    for (std::size_t i = 0; i < monthly_sales.size(); ++i) {
        std::cout << std::setw(4) << (i + 1) << std::setw(6) << monthly_sales[i];
        if (i >= 2) {
            std::cout << std::setw(9) << trend[i - 2];
            if (i >= 3 && trend[i - 2] > trend[i - 3]) {
                std::cout << "    ↑";
            } else if (i >= 3 && trend[i - 2] < trend[i - 3]) {
                std::cout << "    ↓";
            }
        }
        std::cout << std::endl;
    }

    // 自己相関で季節性をチェック
    auto sales_acf = statcpp::acf(monthly_sales.begin(), monthly_sales.end(), 6);
    print_subsection("自己相関分析");
    bool has_pattern = false;
    for (std::size_t i = 1; i < sales_acf.size(); ++i) {
        if (std::abs(sales_acf[i]) > 0.5) {
            std::cout << "  ラグ" << i << "で強い自己相関\n";
            has_pattern = true;
        }
    }
    if (!has_pattern) {
        std::cout << "  弱い自己相関\n";
    }
    std::cout << "→ 上昇トレンドが確認される\n";

    // ============================================================================
    // 10. まとめ：時系列分析のガイドライン
    // ============================================================================
    print_section("まとめ：時系列分析のガイドライン");

    std::cout << R"(
【時系列分析の手順】

1. データの可視化
   - 系列をプロット
   - トレンド、季節性、外れ値を確認

2. 定常性のチェック
   - ACF/PACFプロットを使用
   - 必要に応じて差分を適用

3. パターンの特定
   - ACF: MA次数 (q) の決定
   - PACF: AR次数 (p) の決定
   - 季節パターン（周期）の特定

4. モデルの選択
   - AR: PACFが打ち切れ、ACFが減衰
   - MA: ACFが打ち切れ、PACFが減衰
   - ARMA: 両方が徐々に減衰

5. 予測の評価
   - MAE、RMSE、MAPEを使用
   - 交差検証
   - サンプル外テスト

【よく使う変換】
┌──────────────┬────────────────────────────────┐
│ 変換         │ 目的                           │
├──────────────┼────────────────────────────────┤
│ 差分         │ トレンド除去                   │
├──────────────┼────────────────────────────────┤
│ 季節差分     │ 季節性除去                     │
├──────────────┼────────────────────────────────┤
│ 対数変換     │ 分散の安定化                   │
├──────────────┼────────────────────────────────┤
│ 移動平均     │ ノイズの平滑化                 │
└──────────────┴────────────────────────────────┘

【時系列モデルの種類】
- AR（自己回帰）: 過去の値から予測
- MA（移動平均）: 過去の誤差から予測
- ARMA: ARとMAの組み合わせ
- ARIMA: ARMAに差分を追加
- SARIMA: 季節性を考慮

【実用上のヒント】
1. まず単純なモデルから試す
2. 残差診断を必ず実施
3. 複数モデルを比較
4. ドメイン知識を活用
5. 過学習に注意

【応用分野】
- 金融: 株価予測、リスク管理
- 経済: GDP予測、需要予測
- 気象: 天気予報、気候分析
- 工学: 信号処理、異常検知
- ビジネス: 売上予測、在庫管理
)";

    return 0;
}
