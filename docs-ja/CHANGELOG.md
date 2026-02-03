# 変更履歴

statcpp ライブラリの変更履歴を記録します。

このプロジェクトは [Semantic Versioning](https://semver.org/) に従います。

## [Unreleased]

### 計画中の機能

- より多くの分布関数のサポート
- パフォーマンスの最適化
- より詳細なエラーメッセージ

---

## [0.1.0] - 2024-02-02

### 初回リリース

#### 追加機能

**基本統計量 (Basic Statistics)**
- 合計、平均、中央値、最頻値
- 幾何平均、調和平均、トリム平均
- 重み付き平均、対数平均

**散布度 (Dispersion & Spread)**
- 分散、標準偏差、範囲
- 変動係数、四分位範囲
- 平均絶対偏差

**順序統計量 (Order Statistics)**
- 最小値、最大値
- 四分位数、パーセンタイル
- 五数要約

**分布の形状 (Shape of Distribution)**
- 歪度
- 尖度、超過尖度

**相関・共分散 (Correlation & Covariance)**
- Pearson 相関係数
- Spearman 順位相関係数
- Kendall の順位相関係数
- 共分散

**度数分布 (Frequency Distribution)**
- 度数分布表
- ヒストグラム
- 累積度数分布

**特殊関数 (Special Functions)**
- ガンマ関数、ベータ関数
- 誤差関数
- 正規分布の CDF、PDF、逆関数

**乱数生成 (Random Engine)**
- 一様分布、正規分布の乱数生成

**確率分布 (Probability Distributions)**
- 連続分布: 正規分布、t分布、カイ二乗分布、F分布、指数分布、ガンマ分布、ベータ分布、ワイブル分布、対数正規分布、コーシー分布
- 離散分布: 二項分布、ポアソン分布、幾何分布、負の二項分布、超幾何分布

**統計的推定 (Estimation)**
- 平均、比率、分散の信頼区間

**仮説検定 (Hypothesis Tests)**
- パラメトリック検定: z検定、t検定、F検定、カイ二乗検定
- ノンパラメトリック検定: Wilcoxon検定、Mann-Whitney U検定、Kruskal-Wallis検定、Friedman検定

**効果量 (Effect Size)**
- Cohen's d, Hedges' g, Glass's Δ
- イータ二乗、オメガ二乗
- Cohen's f, R²

**リサンプリング (Resampling)**
- ブートストラップ
- ジャックナイフ
- 置換検定

**検出力分析 (Power Analysis)**
- t検定のサンプルサイズ計算と検出力
- 比率検定のサンプルサイズ計算と検出力

**回帰分析 (Regression)**
- 単回帰、重回帰分析
- 一般化線形モデル (GLM)

**分散分析 (ANOVA)**
- 一元配置分散分析
- 二元配置分散分析
- 反復測定分散分析

**多変量解析・クラスタリング**
- 多変量解析の基本機能
- k-means クラスタリング
- 階層的クラスタリング

**距離・類似度 (Distance & Similarity)**
- ユークリッド距離、マンハッタン距離、チェビシェフ距離
- ミンコフスキー距離
- コサイン類似度、Jaccard係数
- ハミング距離

**数値計算ユーティリティ (Numerical Utilities)**
- 浮動小数点数の近似等値判定
- Kahan 加算アルゴリズム
- 数値安定な対数・指数関数
- 値のクランプと範囲判定

**その他のモジュール**
- 時系列分析
- カテゴリカルデータ分析
- 生存時間解析
- 頑健統計
- データラングリング（欠損値処理を含む）

#### ドキュメント

- Doxygen による完全な API ドキュメント
- 使い方ガイド
- 各モジュールのサンプルプログラム（30以上）
- インストールガイド
- ビルドとテストガイド
- コントリビューションガイド

#### テスト

- Google Test による包括的なテストスイート
- 各モジュールの単体テスト
- エッジケースと例外のテスト

#### その他

- C++17 ヘッダーオンリーライブラリ
- CMake サポート
- クロスプラットフォーム対応（macOS、Linux、Windows）
- MIT ライセンス

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

[Unreleased]: https://github.com/yourusername/statcpp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/statcpp/releases/tag/v0.1.0
