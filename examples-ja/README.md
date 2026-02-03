# statcpp Examples

このディレクトリには、statcppライブラリの各ヘッダーファイルの使用例が含まれています。

## ビルド方法

### CMakeを使用する場合（推奨）

```bash
# プロジェクトルートから
mkdir build
cd build
cmake ..
make
```

サンプルプログラムはgoogleTestと一緒にビルドされ、`build/test/`ディレクトリに生成されます。

### 個別にコンパイルする場合

```bash
g++ -std=c++17 -I../include -o example_basic_statistics example_basic_statistics.cpp
./example_basic_statistics
```

## サンプルプログラム一覧

### Module 1: 記述統計 (Descriptive Statistics)

- **example_basic_statistics.cpp** - 基本統計量（平均、中央値、最頻値など）
- **example_dispersion_spread.cpp** - 散布度（分散、標準偏差、IQRなど）
- **example_order_statistics.cpp** - 順序統計量（最小値、最大値、分位点など）
- **example_shape_of_distribution.cpp** - 分布の形状（歪度、尖度）
- **example_correlation_covariance.cpp** - 相関・共分散（Pearson, Spearman, Kendallなど）
- **example_frequency_distribution.cpp** - 度数分布

### Module 2: 確率分布 (Probability Distributions)

- **example_continuous_distributions.cpp** - 連続分布（正規分布、t分布、χ²分布、F分布、対数正規分布、Weibull分布など）
- **example_discrete_distributions.cpp** - 離散分布（二項分布、ポアソン分布、ベルヌーイ分布、離散一様分布など）

### Module 3: 推測統計 (Inferential Statistics)

- **example_estimation.cpp** - 推定（信頼区間、標準誤差、誤差マージンなど）
- **example_parametric_tests.cpp** - パラメトリック検定（t検定、χ²検定、F検定など）
- **example_nonparametric_tests.cpp** - ノンパラメトリック検定（Wilcoxon検定、Mann-Whitney U検定など）
- **example_resampling.cpp** - リサンプリング（ブートストラップ、置換検定、交差検証など）

### Module 4: 統計モデリング (Statistical Modeling)

- **example_linear_regression.cpp** - 線形回帰（OLS、VIF、残差診断など）
- **example_anova.cpp** - 分散分析（一元配置、二元配置、事後検定など）

### Module 5: 応用分析 (Applied Analysis)

- **example_distance_metrics.cpp** - 距離・類似度メトリクス（ユークリッド距離、マンハッタン距離、コサイン類似度など）

### Module 8: 開発基盤 (Development Infrastructure)

- **example_numerical_utils.cpp** - 数値計算ユーティリティ（浮動小数点比較、収束判定、Kahan加算など）

## 実行例

```bash
# build/testディレクトリから実行
cd build/test

# 基本統計量のサンプル
./example_basic_statistics

# 連続分布のサンプル
./example_continuous_distributions

# 距離メトリクスのサンプル
./example_distance_metrics

# 数値計算ユーティリティのサンプル
./example_numerical_utils
```

## 注意事項

- 全てのサンプルはC++17標準を使用しています
- ヘッダーオンリーライブラリのため、外部ライブラリへのリンクは不要です
- 乱数を使用するサンプルでは、`statcpp::set_seed()`で乱数シードを設定しています

## 追加情報

各サンプルコードの詳細な説明は、対応するヘッダーファイルのドキュメント（README.md）を参照してください。
