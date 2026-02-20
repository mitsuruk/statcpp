# 変更履歴

statcpp ライブラリの変更履歴を記録します。

このプロジェクトは [Semantic Versioning](https://semver.org/) に従います。

## [0.1.1] - 2026-02-20

### Fixed (修正)

- 統合ヘッダー `statcpp.hpp` に欠落していた 9 モジュールのインクルードを追加: `categorical.hpp`, `clustering.hpp`, `data_wrangling.hpp`, `missing_data.hpp`, `multivariate.hpp`, `power_analysis.hpp`, `robust.hpp`, `survival.hpp`, `time_series.hpp`
  - これらのモジュールは個別ヘッダーとしては利用可能でしたが、`#include <statcpp/statcpp.hpp>` 使用時にインクルードされていませんでした
  - 英語版 (`include/`) および日本語版 (`include-ja/`) の両方の統合ヘッダーを更新

---

## [0.1.0] - 2025-02-19

### 初回リリース

C++17 ヘッダーオンリー統計ライブラリ。758 件の単体テストと R 4.4.2 に対する 167 件の数値検証チェックを含みます。

#### 機能

##### 基本統計量 (`basic_statistics.hpp`)

- 合計、平均、中央値、最頻値
- 幾何平均、調和平均、トリム平均
- 重み付き平均、対数平均
- 全関数に射影 (Projection) オーバーロード

##### 散布度 (`dispersion_spread.hpp`)

- 分散（母集団/標本）、標準偏差、範囲
- 変動係数、四分位範囲
- 平均絶対偏差
- 事前計算済み平均オーバーロード

##### 順序統計量 (`order_statistics.hpp`)

- 最小値、最大値
- 四分位数、パーセンタイル（R type=7 / Excel PERCENTILE.INC 互換）
- 五数要約

##### 分布の形状 (`shape_of_distribution.hpp`)

- 歪度
- 尖度、超過尖度

##### 相関・共分散 (`correlation_covariance.hpp`)

- Pearson 相関係数
- Spearman 順位相関係数
- Kendall の順位相関係数
- 共分散

##### 度数分布 (`frequency_distribution.hpp`)

- 度数分布表
- ヒストグラム
- 累積度数分布

##### 特殊関数 (`special_functions.hpp`)

- ガンマ関数、ベータ関数、対数ガンマ
- 誤差関数
- 不完全ベータ/ガンマ関数とその逆関数

##### 乱数生成 (`random_engine.hpp`)

- スレッドセーフなデフォルト乱数エンジン
- シード設定ユーティリティ

##### 確率分布

- 連続分布 (`continuous_distributions.hpp`): 正規分布、t分布、カイ二乗分布、F分布、指数分布、ガンマ分布、ベータ分布、ワイブル分布、対数正規分布、コーシー分布、Studentized range 分布
  - 各分布に CDF、PDF、分位点、乱数生成
  - Studentized range CDF は Copenhaver & Holland (1988) アルゴリズム（R の `ptukey` と同等）
  - Newton-Raphson 分位点関数に数値安定性のための `isfinite` ガード
- 離散分布 (`discrete_distributions.hpp`): 二項分布、ポアソン分布、幾何分布、負の二項分布、超幾何分布

##### 統計的推定 (`estimation.hpp`)

- 平均、比率、分散の信頼区間

##### 仮説検定

- パラメトリック検定 (`parametric_tests.hpp`): z検定、t検定（1標本/2標本/対応あり）、F検定、カイ二乗検定
- ノンパラメトリック検定 (`nonparametric_tests.hpp`): Wilcoxon 符号順位検定、Mann-Whitney U 検定、Kruskal-Wallis 検定、Friedman 検定、Kolmogorov-Smirnov 検定、Levene 検定、Bartlett 検定
  - Wilcoxon、Mann-Whitney U、Kruskal-Wallis のタイ補正

##### 効果量 (`effect_size.hpp`)

- Cohen's d, Hedges' g, Glass's delta
- イータ二乗、オメガ二乗
- Cohen's f, R二乗

##### リサンプリング (`resampling.hpp`)

- ブートストラップ（パーセンタイル法、BCa 法）
- ジャックナイフ
- 置換検定

##### 検出力分析 (`power_analysis.hpp`)

- t検定のサンプルサイズ計算と検出力（1標本/2標本）
- 比率検定のサンプルサイズ計算と検出力
- 文字列ベースと `alternative_hypothesis` 列挙型の両方のオーバーロード

##### 回帰分析

- 単回帰・重回帰分析 (`linear_regression.hpp`)
- 一般化線形モデル (`glm.hpp`): Gaussian、Binomial、Poisson、Gamma 族。Identity、Logit、Probit、Log、Inverse、Cloglog リンク関数
  - IRLS アルゴリズムによる収束追跡と NaN 安全な出力

##### 分散分析 (`anova.hpp`)

- 一元配置分散分析（退化ケース処理付き）
- 二元配置分散分析
- 反復測定分散分析
- 事後検定: Tukey HSD（真の Studentized range）、Bonferroni、Dunnett、Scheffe
  - 全事後検定関数に `se == 0` 退化処理

##### モデル選択 (`model_selection.hpp`)

- AIC、BIC、自由度調整済み R 二乗
- クロスバリデーション
- ステップワイズ選択

##### 多変量解析 (`multivariate.hpp`)

- 主成分分析 (PCA)
- 基本的な多変量関数

##### クラスタリング (`clustering.hpp`)

- k-means クラスタリング
- 階層的クラスタリング（単連結、完全連結、平均連結、Ward 法）

##### 時系列分析 (`time_series.hpp`)

- ACF、PACF
- 移動平均、指数平滑化
- 基本的な時系列操作

##### カテゴリカルデータ分析 (`categorical.hpp`)

- カイ二乗独立性検定
- Fisher の正確検定

##### 生存時間解析 (`survival.hpp`)

- Kaplan-Meier 推定量
- ログランク検定

##### 頑健統計 (`robust.hpp`)

- 中央絶対偏差 (MAD)
- トリム/ウィンソライズ統計量

##### データラングリング (`data_wrangling.hpp`, `missing_data.hpp`)

- NaN 除去、データフィルタリング
- 欠損値処理

##### 距離・類似度 (`distance_metrics.hpp`)

- ユークリッド距離、マンハッタン距離、チェビシェフ距離、ミンコフスキー距離
- コサイン類似度、Jaccard 係数、ハミング距離

##### 数値計算ユーティリティ (`numerical_utils.hpp`)

- 浮動小数点数の近似等値判定
- Kahan 加算アルゴリズム
- 数値安定な対数・指数関数
- 値のクランプと範囲判定

#### アーキテクチャ

- C++17 ヘッダーオンリーライブラリ（ビルド不要）
- STL コンテナ互換のイテレータベースインターフェース
- ヘッダーは `include/statcpp/` に配置、`#include "statcpp/module.hpp"` 形式
- バイリンガルヘッダー: 英語版 (`include/`)、日本語版 (`include-ja/`)
- CMake サポート（`find_package(statcpp)` および `add_subdirectory`）
- クロスプラットフォーム: macOS (Apple Clang)、Linux (GCC)、Windows (MSVC)
- MIT ライセンス

#### ドキュメント

- Doxygen による完全な API ドキュメント
- インクルードパス設定を含む使い方ガイド
- 各モジュールのサンプルプログラム（30 以上）
- インストール、ビルド、コントリビューションガイド

#### テスト

- Google Test による 758 件の単体テスト
- R 4.4.2 に対する 167 件の数値検証チェック
- エッジケースと例外のカバレッジ

---

## 変更履歴のフォーマット

今後のリリースでは、以下の形式で変更を記録します:

### Added (追加)

新しい機能の追加

### Changed (変更)

既存機能の変更

### Deprecated (非推奨)

今後削除予定の機能

### Removed (削除)

削除された機能

### Fixed (修正)

バグ修正

### Security (セキュリティ)

セキュリティに関する変更

---

[0.1.1]: https://github.com/yourusername/statcpp/releases/tag/v0.1.1
[0.1.0]: https://github.com/yourusername/statcpp/releases/tag/v0.1.0
