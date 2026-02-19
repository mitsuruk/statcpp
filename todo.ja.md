# C++向け統計パッケージ作成プロジェクト

本ドキュメントは実装の依存関係を考慮し、開発に適した順序でブロックを配置している。

---

## Module 1: 記述統計 (Descriptive Statistics)

イテレータペアで動作する純粋な統計関数群。他の機能の基盤となる。

### 基本統計量 (Basic Statistics)

- [x] 合計 (Sum)
- [x] データ数 (Count / N)
- [x] 平均値 (Mean) — 算術平均 (Arithmetic Mean)
- [x] 中央値 (Median)
- [x] 最頻値 (Mode)
- [x] 幾何平均 (Geometric Mean)
- [x] 調和平均 (Harmonic Mean)
- [x] トリム平均 (Trimmed Mean)
- [x] 重み付き平均 (Weighted Mean)
- [x] 対数平均 (Logarithmic Mean)
- [x] 重み付き調和平均 (Weighted Harmonic Mean)

### 散布度 (Dispersion & Spread)

- [x] 範囲 (Range) — 最大値 - 最小値
- [x] 分散 (Variance) — 母分散 / 標本分散
- [x] 標準偏差 (Standard Deviation) — 母標準偏差 / 標本標準偏差
- [x] 変動係数 (Coefficient of Variation)
- [x] 四分位範囲 (Interquartile Range, IQR)
- [x] 平均偏差 (Mean Absolute Deviation)
- [ ] 分散の自由度補正オプション (ddof parameter)
- [x] 重み付き分散 (Weighted Variance)
- [x] 幾何標準偏差 (Geometric Standard Deviation)
- [ ] 対数標準偏差 (Log Standard Deviation)

### 順序統計量 (Order Statistics)

- [x] 最小値 (Minimum)
- [x] 最大値 (Maximum)
- [x] 四分位数 (Quartiles — Q1, Q2, Q3)
- [x] パーセンタイル (Percentiles)
- [x] 五数要約 (Five-Number Summary)
- [x] argmin / argmax (インデックス取得)
- [ ] 分位点の補間法指定 (Quantile interpolation methods)
- [x] 重み付きパーセンタイル (Weighted Percentiles)
- [x] 重み付き中央値 (Weighted Median)
- [ ] デシル (Deciles)
- [ ] パーセンタイルランク (Percentile Rank)

### 分布の形状 (Shape of Distribution)

- [x] 歪度 (Skewness)
- [x] 尖度 (Kurtosis)
- [ ] 歪度・尖度のバイアス補正 (Bias-corrected Skewness/Kurtosis)
- [ ] 中心モーメント (Central Moments)
- [ ] 高次統計量 (Higher-order Statistics)

### 相関・共分散 (Correlation & Covariance)

- [x] 共分散 (Covariance)
- [x] ピアソン相関係数 (Pearson Correlation Coefficient)
- [x] スピアマン順位相関係数 (Spearman's Rank Correlation)
- [x] ケンドール順位相関係数 (Kendall's Tau)
- [ ] 偏相関 (Partial Correlation)
- [x] 重み付き共分散 (Weighted Covariance)
- [ ] 重み付き相関行列 (Weighted Correlation Matrix)

### 度数分布 (Frequency Distribution)

- [x] 度数表 (Frequency Table)
- [x] 相対度数 (Relative Frequency)
- [x] 累積度数 (Cumulative Frequency)
- [x] 累積相対度数 (Cumulative Relative Frequency)
- [ ] ヒストグラム計算 (Histogram Computation)
- [ ] ビン統計 (Binned Statistics)
- [ ] 自動ビン決定 (Automatic Binning: Scott/Freedman-Diaconis)

### ユーティリティ関数 (Utility Functions)

- [ ] skip-NaN 集約 (NaN-aware Aggregations)
- [ ] 部分選択 (破壊的/非破壊的) (Partial Selection)
- [ ] カーネル密度推定 (Kernel Density Estimation, KDE)

**Module 1 未実装項目の理由**:
- **分散の自由度補正オプション (ddof)**: 既存の`population_variance`と`sample_variance`で対応可能。追加APIは冗長。
- **対数標準偏差**: 幾何標準偏差で類似機能を提供済み。使用頻度が低い。
- **分位点の補間法指定**: 現在の線形補間（R type=7相当）で十分実用的。複数の補間法をサポートするとAPIが複雑化。
- **デシル、パーセンタイルランク**: `percentile`関数で実現可能。専用関数は不要。
- **バイアス補正歪度・尖度、中心モーメント、高次統計量**: 使用頻度が低く、実装の優先度が低い。
- **偏相関**: 線形回帰の残差相関で代替可能。専用実装は複雑。
- **重み付き相関行列**: 個別の重み付き共分散から構築可能。
- **ヒストグラム計算、ビン統計、自動ビン決定**: 可視化寄りの機能。ヘッダーオンリーライブラリの範囲外。
- **skip-NaN 集約**: C++では明示的なNaNチェックをユーザー側で実施する設計。
- **部分選択**: `std::nth_element`等の標準アルゴリズムで対応可能。
- **カーネル密度推定 (KDE)**: 複雑なアルゴリズム。外部ライブラリ推奨。

---

## Module 2: 確率分布 (Probability Distributions)

検定・推定の基盤。各分布について pdf/pmf, cdf, quantile, rng の4機能を実装対象とする。

### 開発基盤: 特殊関数 (Special Functions)

確率分布の実装に必要な特殊関数を先に整備する。

- [x] ガンマ関数 / 対数ガンマ関数 (Gamma / Log-Gamma Function)
- [x] ベータ関数 / 不完全ベータ関数 (Beta / Incomplete Beta Function)
- [x] 誤差関数 (Error Function, erf/erfc)
- [x] 正規分布関連（Φ, Φ⁻¹）(Normal CDF & Quantile)
- [ ] その他の特殊関数 (Additional Special Functions)

### 乱数生成基盤 (Random Number Generation)

- [x] 乱数エンジン（seed と再現性）(RNG Engine (Seed & Reproducibility))

### 連続分布 (Continuous Distributions)

実装順序: 基本的な分布から、他の分布の基盤となるものを優先

1. - [x] 一様分布 (Uniform Distribution) — pdf / cdf / quantile / rng
2. - [x] 正規分布 (Normal Distribution) — pdf / cdf / quantile / rng
3. - [x] 指数分布 (Exponential Distribution) — pdf / cdf / quantile / rng
4. - [x] ガンマ分布 (Gamma Distribution) — pdf / cdf / quantile / rng
5. - [x] ベータ分布 (Beta Distribution) — pdf / cdf / quantile / rng
6. - [x] χ² 分布 (Chi-square Distribution) — pdf / cdf / quantile / rng ※ガンマ分布の特殊ケース
7. - [x] t 分布 (t-Distribution) — pdf / cdf / quantile / rng
8. - [x] F 分布 (F-Distribution) — pdf / cdf / quantile / rng
9. - [x] 対数正規分布 (Log-normal Distribution) — pdf / cdf / quantile / rng
10. - [x] ワイブル分布 (Weibull Distribution) — pdf / cdf / quantile / rng
11. - [ ] ロジスティック分布 (Logistic Distribution) — pdf / cdf / quantile / rng
12. - [ ] ラプラス分布 (Laplace Distribution) — pdf / cdf / quantile / rng
13. - [ ] コーシー分布 (Cauchy Distribution) — pdf / cdf / quantile / rng
14. - [ ] パレート分布 (Pareto Distribution) — pdf / cdf / quantile / rng
15. - [ ] 逆ガンマ分布 (Inverse Gamma Distribution) — pdf / cdf / quantile / rng
16. - [ ] グンベル分布（極値分布 I 型）(Gumbel Distribution) — pdf / cdf / quantile / rng
17. - [ ] 三角分布 (Triangular Distribution) — pdf / cdf / quantile / rng
18. - [ ] エルラン分布 (Erlang Distribution) — pdf / cdf / quantile / rng

### 離散分布 (Discrete Distributions)

1. - [x] 二項分布 (Binomial Distribution) — pmf / cdf / quantile / rng
2. - [x] ポアソン分布 (Poisson Distribution) — pmf / cdf / quantile / rng
3. - [x] 幾何分布 (Geometric Distribution) — pmf / cdf / quantile / rng
4. - [x] 負の二項分布 (Negative Binomial Distribution) — pmf / cdf / quantile / rng
5. - [x] 超幾何分布 (Hypergeometric Distribution) — pmf / cdf / quantile / rng
6. - [x] ベルヌーイ分布 (Bernoulli Distribution) — pmf / cdf / quantile / rng
7. - [ ] カテゴリカル分布 (Categorical Distribution) — pmf / cdf / quantile / rng
8. - [ ] 多項分布 (Multinomial Distribution) — pmf / cdf / quantile / rng
9. - [x] 離散一様分布 (Discrete Uniform Distribution) — pmf / cdf / quantile / rng

### 多変量分布 (Multivariate Distributions)

- [ ] 多変量正規分布 (Multivariate Normal Distribution)
- [ ] 多変量 t 分布 (Multivariate Student's t Distribution)
- [ ] ディリクレ分布 (Dirichlet Distribution)

### 分布の拡張機能 (Distribution Extensions)

- [ ] 対数尤度（logpdf/logpmf）API (Log-likelihood APIs)
- [ ] 生存関数 (Survival Function)
- [ ] ハザード関数 (Hazard Function)
- [ ] 分布の理論量（平均・分散・歪度・尖度）(Theoretical Moments)
- [ ] 分布のパラメータ推定 (Parameter Estimation for Distributions)
- [ ] 経験分布 (Empirical Distribution)
- [ ] ディラックデルタ分布 (Dirac Delta Distribution)

**Module 2 未実装項目の理由**:
- **追加の連続分布（ロジスティック、ラプラス、コーシー等）**: 主要な分布は実装済み。追加分布は使用頻度が限定的。
- **カテゴリカル分布、多項分布**: 多次元配列の扱いが必要で、ヘッダーオンリーでは複雑。ベルヌーイ・二項分布で代替可能。
- **多変量分布**: 線形代数ライブラリ（行列演算、コレスキー分解等）が必要。ヘッダーオンリーの範囲を大きく超える。
- **対数尤度 API**: `std::log(pdf(x))`で簡単に実現可能。専用APIは冗長。
- **生存関数、ハザード関数**: 生存時間解析で実装済み（Kaplan-Meier等）。分布関数としての汎用実装は優先度低。
- **分布の理論量**: 各分布の平均・分散は数式が既知。関数化する価値は限定的。
- **パラメータ推定**: 最尤推定、モーメント法は複雑な最適化が必要。Phase 3-4の範囲。
- **経験分布**: データから直接計算可能。専用の分布クラスは不要。
- **ディラックデルタ分布**: 理論的な概念。数値計算での実用性が低い。
- **その他の特殊関数**: 現在実装されている特殊関数で主要な分布をカバー済み。

---

## Module 3: 推測統計 (Inferential Statistics)

確率分布に依存。統計的検定と推定。

### 推定 (Estimation)

- [x] 標準誤差（推定）(Standard Error Estimation)
- [x] 信頼区間（平均・比率・分散など）(Confidence Intervals (Mean/Proportion/Variance, etc.))
- [x] 平均の誤差マージン (Mean Margin of Error)
- [x] 比率の誤差マージン (Proportion Margin of Error)
- [x] 最悪ケースの比率誤差マージン (Worst-case Proportion Margin of Error)
- [x] 2標本平均差の推定 (Two-sample Mean Difference)
- [x] 2標本比率差の推定 (Two-sample Proportion Difference)
- [ ] 最尤推定 (Maximum Likelihood Estimation, MLE)
- [ ] モーメント法 (Method of Moments)

### パラメトリック検定 (Parametric Tests)

- [x] z 検定（平均・比率）(z-tests (Mean/Proportion))
- [x] t 検定（1標本・2標本・対応あり）(t-tests (1-sample/2-sample/paired))
- [x] χ² 検定（適合度・独立性）(Chi-square Tests (GOF/Independence))
- [x] F 検定（分散の比較など）(F-tests (Variance Comparison, etc.))
- [x] 比率検定 (Proportion Tests)
- [x] 片側/両側の整理 (One-sided/Two-sided Testing)
- [x] 多重検定補正（Bonferroni, BH など）(Multiple Testing Correction (Bonferroni, BH, etc.))
- [ ] その他の多重検定補正法 (Additional Multiple Testing Corrections)

### ノンパラメトリック検定 (Nonparametric Tests)

- [x] 正規性検定（Shapiro–Wilk、Anderson–Darling、KS など）(Normality Tests)
- [x] 等分散性検定（Levene、Bartlett など）(Homoscedasticity Tests)
- [x] Wilcoxon 符号付順位検定 (Wilcoxon Signed-rank Test)
- [x] Mann–Whitney U 検定 (Mann–Whitney U Test)
- [x] Kruskal–Wallis 検定 (Kruskal–Wallis Test)
- [x] 順位相関（Spearman/Kendall）の整理 (Rank Correlations (Spearman/Kendall))
- [x] Fisher の正確確率検定 (Fisher's Exact Test)

### 効果量・検出力 (Effect Size & Power)

- [x] 効果量（Cohen's d、Hedges' g、相関の効果量等）(Effect Sizes)
- [x] 検出力（Power）とサンプルサイズ設計（基礎）(Power Analysis & Sample Size Planning)

### リサンプリング (Resampling)

- [x] ブートストラップ（CI・バイアス補正）(Bootstrap (CIs/Bias Correction))
- [x] 置換検定 (Permutation Tests)
- [x] 交差検証（基盤）(Cross-validation (Core Utilities))
- [ ] その他の再標本化手法 (Additional Resampling Methods)

**Module 3 未実装項目の理由**:
- **最尤推定 (MLE)、モーメント法**: 複雑な最適化アルゴリズムが必要。Phase 4のモデリングで部分的に対応済み（GLMなど）。汎用的なMLE実装は範囲外。
- **その他の多重検定補正法**: Bonferroni、Benjamini-Hochbergで主要なケースをカバー済み。追加の補正法は使用頻度が低い。
- **その他の再標本化手法**: ブートストラップ、置換検定、交差検証で主要な手法を実装済み。ジャックナイフ等は類似の概念で代替可能。

---

## Module 4: 統計モデリング (Statistical Modeling)

推測統計に依存。回帰・GLM・ANOVA。

### 線形回帰 (Linear Regression / OLS)

- [x] 最小二乗推定 (Least Squares Estimation)
- [x] 回帰係数の標準誤差と検定 (Coefficient SEs & Tests)
- [x] 決定係数 R² (R-squared)
- [x] 残差診断 (Residual Diagnostics)
- [x] 多重共線性（VIF 等）(Multicollinearity (VIF, etc.))
- [x] 予測区間 (Prediction Intervals)
- [ ] ロバスト回帰（Huber、RANSAC 等）(Robust Regression)

### 分散分析 (ANOVA & Extensions)

- [x] 一元/二元 ANOVA (One-way/Two-way ANOVA)
- [x] ANCOVA (ANCOVA)
- [x] 交互作用 (Interaction Effects)
- [x] 事後比較（Tukey、Dunnett など）(Post-hoc Comparisons)

### 一般化線形モデル (Generalized Linear Models / GLM)

- [x] ロジスティック回帰 (Logistic Regression)
- [x] ポアソン回帰 (Poisson Regression)
- [x] リンク関数と分散関数 (Link & Variance Functions)
- [x] 反復最適化（IRLS 等）(Iterative Optimization (IRLS, etc.))
- [x] 逸脱度と適合度指標 (Deviance & Goodness-of-fit Metrics)
- [ ] ロジスティック回帰の拡張（正則化等）(Extended Logistic Regression)

### モデル選択・正則化 (Model Selection & Regularization)

- [x] モデル選択指標（AIC/BIC/調整済みR²）(Model Selection Criteria)
- [x] 正則化回帰（Ridge/Lasso/Elastic Net）(Regularized Regression)
- [x] 交差検証の具体（k-fold/層化/時系列CV）(Cross-validation Variants)

### 高度なモデリング (Advanced Modeling)

- [ ] 混合効果モデル（LMM/GLMM）(Mixed-effects Models)
- [ ] 一般化加法モデル（GAM）(Generalized Additive Models)
- [x] 多重共線性診断の拡張（条件数など）(Extended Multicollinearity Diagnostics)

### ベイズ推定 (Bayesian Inference) ※任意

- [ ] 共役事前分布（基礎）(Conjugate Priors (Basics))
- [ ] MCMC（導入）(MCMC (Introduction))
- [ ] 変分推論（必要なら）(Variational Inference (Optional))
- [ ] モデル比較 (Model Comparison)
- [ ] 事後確率・ベイズファクター（任意）(Posterior Probability / Bayes Factor)

**Module 4 未実装項目の理由**:
- **ロバスト回帰（Huber、RANSAC等）**: 反復最適化と重み付け再計算が必要。実装が複雑で、外部ライブラリ（例: Ceres Solver）推奨。
- **ロジスティック回帰の拡張（正則化等）**: Ridge/Lasso回帰で正則化の概念を実装済み。ロジスティック回帰への適用は類似の実装で対応可能だが、優先度低。
- **混合効果モデル（LMM/GLMM）**: ランダム効果の推定に複雑な行列演算と最適化が必要。専門ライブラリ（lme4、nlme等）の使用を推奨。
- **一般化加法モデル（GAM）**: スプライン関数の実装と平滑化パラメータの選択が必要。非常に複雑。
- **ベイズ推定（MCMC、変分推論等）**: 大規模な数値計算基盤が必要。Stan、PyMC等の専門ツールに任せるべき。共役事前分布のみの実装も実用性が限定的。

---

## Module 5: 応用分析 (Applied & Domain-Specific Analysis)

特定のデータ構造やドメインに紐づく分析手法。Phase 1-4 の機能を活用。

### 多変量解析 (Multivariate Analysis)

- [x] 共分散/相関行列 (Covariance/Correlation Matrix)
- [x] PCA（主成分分析）(Principal Component Analysis, PCA)
- [ ] 因子分析 (Factor Analysis)
- [ ] 判別分析（必要なら）(Discriminant Analysis (Optional))
- [x] 標準化/スケーリングの統合 (Standardization/Scaling Integration)
- [ ] 標準化・スコアリングの拡張 (Extended Standardization/Scoring)

### 時系列解析 (Time Series Analysis)

- [x] 自己相関/偏自己相関 (ACF/PACF)
- [ ] ARIMA（基礎）(ARIMA (Basics))
- [ ] 季節性の扱い (Seasonality Handling)
- [x] 予測評価（誤差指標: MAE/RMSE/MAPE 等）(Forecast Evaluation)
- [ ] 時系列分解（STL 等）(Time Series Decomposition)
- [ ] ARIMA/季節性の拡張 (Extended ARIMA/Seasonality)

### カテゴリデータ解析 (Categorical Data Analysis)

- [x] 分割表（クロス集計）(Contingency Tables)
- [x] オッズ比・相対リスク (Odds Ratio / Relative Risk)
- [ ] ロジ線形モデル（必要なら）(Log-linear Models (Optional))

### 生存時間解析 (Survival Analysis)

- [x] Kaplan–Meier 推定 (Kaplan–Meier Estimator)
- [x] log-rank 検定 (Log-rank Test)
- [ ] Cox 比例ハザード回帰 (Cox Proportional Hazards Model)
- [ ] 生存解析の拡張機能 (Extended Survival Analysis)

### ロバスト統計・診断 (Robust Statistics & Diagnostics)

- [x] 中央絶対偏差（MAD; Median Absolute Deviation）(Median Absolute Deviation)
- [x] ロバスト推定量（中央値等）の拡張 (Robust Estimators Extensions)
- [ ] ロバスト回帰（Huber、RANSAC 等）(Robust Regression (Optional))
- [x] 影響度指標（Cook 距離等）(Influence Measures (Cook's Distance, etc.))
- [x] 外れ値検出（IQR/Tukey、LOF など）(Outlier Detection)
- [ ] トリミング・外れ値除去の拡張 (Extended Trimming/Outlier Removal)

### クラスタリング・次元削減 (Clustering & Dimensionality Reduction)

- [x] クラスタリング（k-means、階層、DBSCAN 等）(Clustering)
- [ ] 次元削減（t-SNE/UMAP; 任意）(Dimensionality Reduction (Optional))

### 距離・類似度 (Distance & Similarity Metrics)

- [x] ユークリッド距離 (Euclidean Distance)
- [x] マンハッタン距離 (Manhattan Distance)
- [x] マハラノビス距離 (Mahalanobis Distance)
- [x] コサイン類似度 (Cosine Similarity)
- [x] その他の統計的距離 (Minkowski, Chebyshev)

### 情報理論 (Information Theory)

- [ ] エントロピー (Entropy)
- [ ] 相互情報量 (Mutual Information)
- [ ] KLダイバージェンス (Kullback-Leibler Divergence)
- [ ] JSダイバージェンス (Jensen-Shannon Divergence)
- [ ] 情報理論量の差分 (Information-theoretic Differences)

### 方向統計 (Directional Statistics)

- [ ] 円形平均 (Circular Mean)
- [ ] 円形分散 (Circular Variance)
- [ ] von Mises 分布 (von Mises Distribution)

**Module 5 未実装項目の理由**:
- **因子分析**: 固有値分解と因子回転が必要。線形代数ライブラリへの依存が大きい。PCAで類似の次元削減が可能。
- **判別分析**: 線形判別分析（LDA）は線形代数が必要。ロジスティック回帰で代替可能。
- **標準化・スコアリングの拡張**: 基本的な標準化は実装済み。追加のスケーリング手法（Robust Scaler等）は優先度低。
- **ARIMA（基礎）、季節性の扱い、時系列分解**: 非常に複雑なアルゴリズム。専門ライブラリ（statsmodels、forecast等）推奨。
- **ロジ線形モデル**: GLMの一種として実装可能だが、カテゴリカルデータの扱いが複雑。使用頻度も限定的。
- **Cox比例ハザード回帰**: 部分尤度の最適化が必要。生存時間解析の基礎（Kaplan-Meier、log-rank検定）は実装済み。Cox回帰は専門的すぎる。
- **生存解析の拡張機能**: Kaplan-Meier推定量とlog-rank検定で基本的な解析が可能。競合リスク分析等の拡張は専門的。
- **ロバスト回帰**: Phase 4と同じ理由。反復最適化が必要で複雑。
- **トリミング・外れ値除去の拡張**: 基本的な外れ値検出（IQR法、MAD等）は実装済み。追加手法は優先度低。
- **次元削減（t-SNE/UMAP）**: 非線形次元削減は非常に複雑。反復最適化と近傍探索が必要。専門ライブラリ推奨。
- **エントロピー、相互情報量、KLダイバージェンス、JSダイバージェンス**: 離散確率分布の場合は実装可能だが、連続分布では数値積分が必要。優先度は中程度だが、今回は距離メトリクスを優先。
- **情報理論量の差分**: 基本的な情報理論量が未実装のため、派生機能も未実装。
- **方向統計（円形平均、円形分散、von Mises分布）**: 特殊なドメイン（角度データ）向け。使用頻度が非常に限定的。

---

## Module 6: データ基盤 (Data Infrastructure)

### データ構造 (Data Structures)

- [ ] データコンテナ設計（Series / DataFrame 相当）(Data Container Design)
- [ ] 型システム（数値・カテゴリ・日時・欠損）(Type System)
- [ ] 型推論とスキーマ定義 (Type Inference & Schema Definition)
- [ ] インデックス/ラベル（row/column 名）(Indexing & Labels)

### 入出力 (I/O)

- [ ] CSV/TSV 入出力 (CSV/TSV I/O)
- [ ] JSON 入出力 (JSON I/O)
- [ ] バイナリ形式（Parquet 等）の検討 (Columnar/Binary Formats)

### データ前処理 (Data Wrangling & Preprocessing)

- [x] 欠損値処理（除外・代入）(Missing Data Handling (Drop/Impute))
- [x] 外れ値処理（検出・除外・Winsorize）(Outlier Handling) ※ robust.hpp で実装済み
- [x] フィルタ・抽出 (Filtering)
- [x] 変換・派生列の作成 (Transformations & Derived Columns)
- [x] グループ化と集約 (Group-by & Aggregation)
- [ ] 結合（Join / Merge）(Joins / Merges)
- [ ] リシェイプ（wide/long）(Reshaping (Wide/Long))
- [x] ソート（単一キー/複合キー、安定ソート）(Sorting)
- [x] サンプリング（無作為/層化/重複あり・なし）(Sampling)
- [x] 重複行の検出・除去 (Duplicate Detection & Deduplication)
- [x] ローリング/ウィンドウ集計（移動平均・移動分散など）(Rolling/Window Aggregations)
- [x] カテゴリのエンコード（one-hot/ordinal 等）(Categorical Encoding)
- [x] 基本的なデータ検証（範囲・型・欠損率など）(Data Validation)

### 欠損データの高度処理 (Advanced Missing Data)

- [x] MCAR/MAR/MNAR の整理 (MCAR/MAR/MNAR Taxonomy)
- [x] 多重代入 (Multiple Imputation)
- [x] 感度分析 (Sensitivity Analysis)

**Module 6 未実装項目の理由**:
統計関数とは独立して開発可能。大規模な設計が必要なため、コア機能完成後に着手。

---

## Module 7: 可視化 (Visualization)


### 探索的データ分析向け可視化 (EDA Visualization)

- [ ] ヒストグラム（ビン決定: Scott/Freedman–Diaconis 等）(Histogram)
- [ ] 箱ひげ図 (Box Plot)
- [ ] QQ プロット (Q-Q Plot)
- [ ] 散布図 (Scatter Plot)
- [ ] 散布図行列 (Scatterplot Matrix)
- [ ] 相関ヒートマップ (Correlation Heatmap)
- [ ] カテゴリ別比較（棒・バイオリン等）(Category-wise Comparison)

### 診断プロット (Diagnostic Plots)

- [ ] 回帰診断プロット（残差 vs 予測、QQ、影響度）(Regression Diagnostic Plots)
- [ ] ACF/PACF プロット (ACF/PACF Plots)
- [ ] ROC 曲線・AUC（分類モデルが入る場合）(ROC Curve & AUC)

**Module 7 未実装項目の理由**:
他の機能が揃ってから着手。外部ライブラリとの連携も検討。

---

## Module 8: 開発基盤 (Development Infrastructure)

各 Module と並行して整備。

### 数値計算基盤 (Numerical & Optimization Core)

- [ ] 最適化（勾配・ヘッセ・収束判定）(Optimization)
- [ ] 線形代数（分解・安定化）(Linear Algebra)
- [x] 数値安定性と精度設計 (Numerical Stability & Precision Design)
- [ ] 数値積分（CDF/期待値計算のため）(Numerical Integration)
- [ ] 数値微分と自動微分（任意）(Numerical/Automatic Differentiation)
- [ ] 性能最適化（SIMD/並列化の検討）(Performance Optimization)
- [x] 精度・収束ユーティリティ (Precision & Convergence Utilities)

### 再現性・レポーティング (Reproducibility & Reporting)

- [ ] 乱数と環境情報の固定 (RNG & Environment Capture)
- [ ] 分析ログ（パラメータ・バージョン）(Analysis Logging)
- [ ] 結果オブジェクトのシリアライズ (Result Object Serialization)
- [ ] 表/レポート向け整形出力 (Formatted Output for Tables/Reports)

### API設計・相互運用 (API Design & Interoperability)

- [ ] パイプライン/チェーン API (Pipeline/Chaining API)
- [ ] エラーモデルと例外設計 (Error Model & Exception Design)
- [ ] 拡張点（プラグイン設計）(Extensibility (Plugin Architecture))
- [ ] 他言語バインディング（必要なら）(Language Bindings (Optional))

### テスト・ベンチマーク (Testing & Benchmarking)

- [ ] テスト（既知値・境界値・ランダムテスト）(Testing)
- [ ] ベンチマーク（アルゴリズム別の性能比較）(Benchmarking)

**Module 8 未実装項目の理由**:

**数値計算基盤**:
- **最適化（勾配・ヘッセ・収束判定）**: 複雑な最適化アルゴリズム（BFGS、L-BFGS等）は外部ライブラリ（Ceres、NLopt等）推奨。汎用的な実装はヘッダーオンリーの範囲を大きく超える。
- **線形代数（分解・安定化）**: LU分解、QR分解、SVD等は大規模な実装が必要。Eigen、Armadillo等の専門ライブラリ使用を推奨。
- **数値積分**: 既存の特殊関数（不完全ベータ関数、ガンマ関数等）で主要な確率分布のCDFを実装済み。追加の汎用数値積分は優先度低。
- **数値微分と自動微分**: 有限差分法は簡単に実装可能だが精度が低い。自動微分はテンプレートメタプログラミングで実装可能だが非常に複雑。外部ライブラリ（autodiff、CppAD等）推奨。
- **性能最適化（SIMD/並列化）**: コンパイラ依存、プラットフォーム依存の実装が必要。ヘッダーオンリーライブラリでは保守性が低下。最適化はコンパイラに任せる方針。

**再現性・レポーティング**:
- **乱数と環境情報の固定**: `random_engine.hpp`で乱数シード設定が可能。環境情報（OS、コンパイラバージョン等）のキャプチャはヘッダーオンリーでは困難。
- **分析ログ、結果のシリアライズ**: ファイルI/Oが必要。ヘッダーオンリーライブラリの範囲外。ユーザー側で実装すべき機能。
- **表/レポート向け整形出力**: 統計結果の表示形式はユーザーのニーズに大きく依存。標準的なフォーマットを強制するよりも、計算結果の構造体を返してユーザーが自由にフォーマットする方が柔軟。

**API設計・相互運用**:
- **パイプライン/チェーン API**: 現在のイテレータベースAPIを大幅に変更する必要がある。既存の設計との互換性が失われる。Ranges library（C++20以降）の普及を待つべき。
- **エラーモデルと例外設計**: 全関数で`std::invalid_argument`による例外処理を統一実装済み。追加の設計は不要。
- **拡張点（プラグイン設計）**: ヘッダーオンリーライブラリではプラグイン機構の実装が困難。ユーザーは既存の関数を組み合わせて独自の分析を実装可能。
- **他言語バインディング**: Python（pybind11）、R（Rcpp）等のバインディングは別プロジェクトとして開発すべき。statcpp自体はC++ライブラリとして完結。

**テスト・ベンチマーク**:
- **テスト（既知値・境界値・ランダムテスト）**: Phase 1-5と8で各機能のテストコードを作成済み。既知値、境界値、特殊値（NaN、無限大等）を網羅的にテスト。追加の体系的なテストフレームワークは優先度低。
- **ベンチマーク（アルゴリズム別の性能比較）**: ベンチマークはユーザーの環境・データに依存。ライブラリとして提供する価値は限定的。必要に応じてユーザーが実施すべき。

---

## 実装メモ (Implementation Notes)

### 優先度の考え方

1. **Phase 1-3**: 統計の基礎。最優先で実装
2. **Phase 4**: モデリング。Phase 1-3に依存
3. **Phase 5**: 応用分析。特定ドメイン向け
4. **Phase 6-7**: データ基盤と可視化。大規模設計が必要
5. **Phase 8**: 開発基盤。全体を通じて継続的に整備

### 各Phaseでの追加項目の位置づけ

- **重み付き統計量**: Phase 1に追加（基礎統計の自然な拡張）
- **追加の確率分布**: Phase 2に追加（分布の充実）
- **情報理論・距離**: Phase 5に新セクション追加（応用的）
- **方向統計**: Phase 5に新セクション追加（特殊ドメイン）
- **数値計算基盤の拡張**: Phase 8に統合

### 実装済み項目

Phase 1-5の主要機能は実装完了。現在はサンプルコード作成フェーズ。
