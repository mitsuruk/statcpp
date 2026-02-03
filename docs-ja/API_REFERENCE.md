# APIリファレンス

statcpp ライブラリの全モジュールと主要な関数の概要を説明します。

詳細な関数仕様、数式、使用例については、Doxygen で生成された HTML ドキュメントを参照してください。

## ドキュメント生成

詳細な API ドキュメントを生成するには:

```bash
# Doxygen のインストール（必要な場合）
# macOS
brew install doxygen

# Ubuntu/Debian
sudo apt-get install doxygen

# ドキュメント生成
./generate_docs.sh

# または直接実行
doxygen Doxyfile

# ブラウザで開く (macOS)
open docs/html/index.html

# Linux
xdg-open docs/html/index.html
```

## モジュール一覧

### 1. Basic Statistics (`basic_statistics.hpp`)

基本的な統計量を計算する関数群。

**主要な関数:**
- `sum` - 合計
- `count` - 要素数
- `mean` - 算術平均
- `median` - 中央値（ソート済み範囲が必要）
- `mode` - 最頻値
- `modes` - すべての最頻値
- `geometric_mean` - 幾何平均
- `harmonic_mean` - 調和平均
- `trimmed_mean` - トリム平均（ソート済み範囲が必要）
- `weighted_mean` - 重み付き平均
- `logarithmic_mean` - 対数平均
- `weighted_harmonic_mean` - 重み付き調和平均

### 2. Dispersion & Spread (`dispersion_spread.hpp`)

データの散布度（ばらつき）を測定する関数群。

**主要な関数:**
- `range` - 範囲（最大値 - 最小値）
- `population_variance` - 母分散
- `sample_variance` - 標本分散
- `variance` - 分散（デフォルトは標本分散）
- `population_stddev` - 母標準偏差
- `sample_stddev` - 標本標準偏差
- `stddev` - 標準偏差（デフォルトは標本標準偏差）
- `coefficient_of_variation` - 変動係数
- `iqr` - 四分位範囲（ソート済み範囲が必要）
- `mean_absolute_deviation` - 平均絶対偏差

### 3. Order Statistics (`order_statistics.hpp`)

順序統計量を計算する関数群（すべてソート済み範囲が必要）。

**主要な関数:**
- `minimum` - 最小値
- `maximum` - 最大値
- `quartiles` - 四分位数（Q1, Q2, Q3）
- `percentile` - パーセンタイル
- `five_number_summary` - 五数要約（最小値、Q1、中央値、Q3、最大値）

### 4. Shape of Distribution (`shape_of_distribution.hpp`)

分布の形状を特徴付ける統計量。

**主要な関数:**
- `skewness` - 歪度（分布の非対称性）
- `kurtosis` - 尖度（分布の尖り具合）
- `excess_kurtosis` - 超過尖度（正規分布を基準とした尖度）

### 5. Correlation & Covariance (`correlation_covariance.hpp`)

2変数間の関係を測定する関数群。

**主要な関数:**
- `covariance` - 共分散
- `correlation` - Pearson 相関係数
- `spearman_correlation` - Spearman 順位相関係数
- `kendall_tau` - Kendall の順位相関係数

### 6. Frequency Distribution (`frequency_distribution.hpp`)

度数分布とヒストグラムの作成。

**主要な関数:**
- `frequency_table` - 度数分布表
- `histogram` - ヒストグラム（等間隔ビニング）
- `cumulative_frequency` - 累積度数分布

### 7. Special Functions (`special_functions.hpp`)

統計計算で使用される特殊関数。

**主要な関数:**
- `gamma` - ガンマ関数
- `lgamma` - 対数ガンマ関数
- `beta` - ベータ関数
- `lbeta` - 対数ベータ関数
- `erf` - 誤差関数
- `erfc` - 相補誤差関数
- `normal_cdf` - 標準正規分布の累積分布関数
- `normal_pdf` - 標準正規分布の確率密度関数
- `inverse_normal_cdf` - 標準正規分布の逆累積分布関数

### 8. Random Engine (`random_engine.hpp`)

乱数生成エンジン。

**主要な関数:**
- `RandomEngine` クラス - 乱数生成器のラッパー
  - `uniform` - 一様分布
  - `normal` - 正規分布
  - `seed` - シード設定

### 9. Continuous Distributions (`continuous_distributions.hpp`)

連続確率分布のPDF、CDF、分位点関数、乱数生成。

**主要な分布:**
- 一様分布 (`uniform_*`)
- 正規分布 (`normal_*`)
- t分布 (`t_*`)
- カイ二乗分布 (`chisq_*`)
- F分布 (`f_*`)
- 指数分布 (`exponential_*`)
- ガンマ分布 (`gamma_*`)
- ベータ分布 (`beta_*`)
- ワイブル分布 (`weibull_*`)
- 対数正規分布 (`lognormal_*`)

各分布に対して `_pdf`, `_cdf`, `_quantile`, `_rand` 関数が提供されています。

### 10. Discrete Distributions (`discrete_distributions.hpp`)

離散確率分布のPMF、CDF、分位関数、乱数生成。

**主要な分布:**
- ベルヌーイ分布 (`bernoulli_*`)
- 二項分布 (`binomial_*`)
- ポアソン分布 (`poisson_*`)
- 幾何分布 (`geometric_*`)
- 負の二項分布 (`nbinom_*`)
- 超幾何分布 (`hypergeom_*`)
- 離散一様分布 (`discrete_uniform_*`)

各分布に対して `_pmf`, `_cdf`, `_quantile`, `_rand` 関数が提供されています。

### 11. Estimation (`estimation.hpp`)

統計的推定（信頼区間など）。

**主要な関数:**
- `mean_confidence_interval` - 平均の信頼区間
- `proportion_confidence_interval` - 比率の信頼区間
- `variance_confidence_interval` - 分散の信頼区間

### 12. Parametric Tests (`parametric_tests.hpp`)

パラメトリック仮説検定。

**主要な関数:**
- `z_test_one_sample` - 1標本z検定
- `t_test_one_sample` - 1標本t検定
- `t_test_two_sample` - 2標本t検定（独立）
- `paired_t_test` - 対応のあるt検定
- `f_test` - F検定（等分散性の検定）
- `chi_squared_test` - カイ二乗検定

### 13. Nonparametric Tests (`nonparametric_tests.hpp`)

ノンパラメトリック仮説検定。

**主要な関数:**
- `wilcoxon_signed_rank_test` - Wilcoxon符号順位検定
- `mann_whitney_u_test` - Mann-Whitney U検定
- `kruskal_wallis_test` - Kruskal-Wallis検定
- `friedman_test` - Friedman検定

### 14. Effect Size (`effect_size.hpp`)

効果量の計算。

**主要な関数:**
- `cohens_d_one_sample` - Cohen's d（1標本）
- `cohens_d_two_sample` - Cohen's d（2標本）
- `hedges_g` - Hedges' g
- `glass_delta` - Glass's Δ
- `eta_squared` - イータ二乗（η²）
- `omega_squared` - オメガ二乗（ω²）
- `cohens_f` - Cohen's f
- `r_squared` - 決定係数（R²）

### 15. Resampling (`resampling.hpp`)

リサンプリング法。

**主要な関数:**
- `bootstrap` - ブートストラップ
- `bootstrap_confidence_interval` - ブートストラップ信頼区間
- `jackknife` - ジャックナイフ
- `permutation_test` - 置換検定

### 16. Power Analysis (`power_analysis.hpp`)

検出力分析とサンプルサイズ計算。

**主要な関数:**
- `power_t_test_one_sample` - 1標本t検定の検出力
- `sample_size_t_test_one_sample` - 1標本t検定のサンプルサイズ
- `power_t_test_two_sample` - 2標本t検定の検出力
- `sample_size_t_test_two_sample` - 2標本t検定のサンプルサイズ
- `power_prop_test` - 比率検定の検出力
- `sample_size_prop_test` - 比率検定のサンプルサイズ

### 17. Linear Regression (`linear_regression.hpp`)

線形回帰分析。

**主要な関数:**
- `simple_linear_regression` - 単回帰分析
- `multiple_linear_regression` - 重回帰分析
- `predict` - 予測
- `residuals` - 残差
- `r_squared` - 決定係数

### 18. ANOVA (`anova.hpp`)

分散分析。

**主要な関数:**
- `one_way_anova` - 一元配置分散分析
- `two_way_anova` - 二元配置分散分析
- `repeated_measures_anova` - 反復測定分散分析

### 19. GLM (`glm.hpp`)

一般化線形モデル。

**主要な関数:**
- `logistic_regression` - ロジスティック回帰
- `poisson_regression` - ポアソン回帰
- `glm_fit` - GLMの当てはめ

### 20. Distance & Similarity Metrics (`distance_metrics.hpp`)

距離と類似度の計算。

**主要な関数:**
- `euclidean_distance` - ユークリッド距離
- `manhattan_distance` - マンハッタン距離
- `chebyshev_distance` - チェビシェフ距離
- `minkowski_distance` - ミンコフスキー距離
- `cosine_similarity` - コサイン類似度
- `jaccard_similarity` - Jaccard係数
- `hamming_distance` - ハミング距離

### 21. Numerical Utilities (`numerical_utils.hpp`)

数値計算のユーティリティ関数。

**主要な関数:**
- `approx_equal` - 浮動小数点数の近似等値判定
- `kahan_sum` - Kahan加算アルゴリズム（高精度和）
- `log1p_safe` - log(1 + x) の安定計算
- `expm1_safe` - exp(x) - 1 の安定計算
- `clamp` - 値の範囲制限
- `in_range` - 範囲判定
- `relative_error` - 相対誤差

### その他のモジュール

プロジェクトには以下の追加モジュールも含まれています:

- **Multivariate Analysis** (`multivariate.hpp`) - 多変量解析（共分散行列、PCAなど）
- **Time Series Analysis** (`time_series.hpp`) - 時系列分析（ACF/PACF、移動平均など）
- **Categorical Data Analysis** (`categorical.hpp`) - カテゴリカルデータ分析（分割表、オッズ比など）
- **Survival Analysis** (`survival.hpp`) - 生存時間解析（Kaplan-Meier、log-rank検定など）
- **Robust Statistics** (`robust.hpp`) - 頑健統計（MAD、外れ値検出など）
- **Clustering** (`clustering.hpp`) - クラスタリング（k-means、階層的クラスタリングなど）
- **Data Wrangling** (`data_wrangling.hpp`) - データ変換・前処理
- **Missing Data** (`missing_data.hpp`) - 欠損データの高度な処理（MCAR検定、多重代入法など）
- **Model Selection** (`model_selection.hpp`) - モデル選択（AIC/BIC、Ridge/Lasso/Elastic Net回帰など）

## 共通の設計原則

### イテレータベースのインターフェース

すべての関数は STL スタイルのイテレータペア `(first, last)` を受け取ります。

```cpp
std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
double avg = statcpp::mean(data.begin(), data.end());
```

### 射影サポート

多くの関数は射影関数をサポートし、構造体のメンバーなどを直接計算できます。

```cpp
struct Point { double x, y; };
std::vector<Point> points = {{1, 2}, {3, 4}, {5, 6}};

// x座標の平均
double avg_x = statcpp::mean(points.begin(), points.end(),
                              [](const Point& p) { return p.x; });
```

### 例外処理

不正な入力（空の範囲、範囲外のパラメータなど）に対しては `std::invalid_argument` を投げます。

```cpp
std::vector<double> empty;
try {
    double avg = statcpp::mean(empty.begin(), empty.end());
} catch (const std::invalid_argument& e) {
    std::cerr << e.what() << std::endl;  // "statcpp::mean: empty range"
}
```

## 使用例

### 基本的な統計分析

```cpp
#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"
#include <vector>
#include <algorithm>

std::vector<double> data = {5, 2, 8, 1, 3, 7, 4};

// 基本統計量
double avg = statcpp::mean(data.begin(), data.end());
double sd = statcpp::stddev(data.begin(), data.end());

// 順序統計量（ソートが必要）
std::sort(data.begin(), data.end());
double med = statcpp::median(data.begin(), data.end());
auto q = statcpp::quartiles(data.begin(), data.end());
```

### 仮説検定

```cpp
#include "statcpp/parametric_tests.hpp"

std::vector<double> sample1 = {/* データ */};
std::vector<double> sample2 = {/* データ */};

// 2標本t検定
auto result = statcpp::t_test_two_sample(
    sample1.begin(), sample1.end(),
    sample2.begin(), sample2.end()
);

std::cout << "t統計量: " << result.statistic << std::endl;
std::cout << "p値: " << result.p_value << std::endl;
```

### 線形回帰

```cpp
#include "statcpp/linear_regression.hpp"

std::vector<double> x = {1, 2, 3, 4, 5};
std::vector<double> y = {2, 4, 5, 4, 5};

auto model = statcpp::simple_linear_regression(
    x.begin(), x.end(),
    y.begin()
);

std::cout << "切片: " << model.intercept << std::endl;
std::cout << "傾き: " << model.slope << std::endl;
std::cout << "R²: " << model.r_squared << std::endl;
```

## 次のステップ

- 実用的なコード例は [サンプルコード](EXAMPLES.md) を参照してください
- 基本的な使い方は [使い方ガイド](USAGE.md) を参照してください
- 詳細な関数仕様は Doxygen 生成ドキュメントを参照してください
