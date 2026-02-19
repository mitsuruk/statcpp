# APIリファレンス

statcpp ライブラリの全モジュールと公開関数の概要を説明します（31 個のヘッダーファイルに 524 個の公開関数）。

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

### 1. 基本統計量 (`basic_statistics.hpp`)

基本的な統計量を計算する関数群。

**関数:**

| 関数 | 説明 | オーバーロード |
| ---- | ---- | -------------- |
| `sum` | 合計 | + 射影 |
| `count` | 要素数 | |
| `mean` | 算術平均 | + 射影 |
| `median` | 中央値（ソート済み範囲が必要） | + 射影 |
| `mode` | 最頻値 | + 射影 |
| `modes` | すべての最頻値 | + 射影 |
| `geometric_mean` | 幾何平均 | + 射影 |
| `harmonic_mean` | 調和平均 | + 射影 |
| `trimmed_mean` | トリム平均（ソート済み範囲が必要） | + 射影 |
| `weighted_mean` | 重み付き平均 | + 射影 |
| `logarithmic_mean` | 2値の対数平均 | |
| `weighted_harmonic_mean` | 重み付き調和平均 | + 射影 |
| `argmin` | 最小値のインデックス | + 射影 |
| `argmax` | 最大値のインデックス | + 射影 |

### 2. 散布度 (`dispersion_spread.hpp`)

データの散布度（ばらつき）を測定する関数群。

**関数:**

| 関数 | 説明 | オーバーロード |
| ---- | ---- | -------------- |
| `range` | 範囲（最大値 - 最小値） | + 射影 |
| `var` | ddof パラメータ付き分散 | + 事前計算済み平均, + 射影 |
| `population_variance` | 母分散 | + 事前計算済み平均, + 射影 |
| `sample_variance` | 標本分散 | + 事前計算済み平均, + 射影 |
| `variance` | 分散（デフォルトは標本分散） | + 事前計算済み平均, + 射影 |
| `stdev` | ddof パラメータ付き標準偏差 | + 事前計算済み平均, + 射影 |
| `population_stddev` | 母標準偏差 | + 事前計算済み平均, + 射影 |
| `sample_stddev` | 標本標準偏差 | + 事前計算済み平均, + 射影 |
| `stddev` | 標準偏差（デフォルトは標本標準偏差） | + 事前計算済み平均, + 射影 |
| `coefficient_of_variation` | 変動係数 | + 事前計算済み平均, + 射影 |
| `iqr` | 四分位範囲（ソート済み範囲が必要） | + 射影 |
| `mean_absolute_deviation` | 平均絶対偏差 | + 事前計算済み平均, + 射影 |
| `weighted_variance` | 重み付き分散 | + 射影 |
| `weighted_stddev` | 重み付き標準偏差 | + 射影 |
| `geometric_stddev` | 幾何標準偏差 | + 射影 |

### 3. 順序統計量 (`order_statistics.hpp`)

順序統計量を計算する関数群（すべてソート済み範囲が必要）。

**構造体:**

- `quartile_result` — フィールド: `Q1`, `Q2`, `Q3`
- `five_number_summary_result` — フィールド: `min`, `Q1`, `median`, `Q3`, `max`

**関数:**

| 関数 | 説明 | オーバーロード |
| ---- | ---- | -------------- |
| `interpolate_at` | 小数インデックスでの補間 | + 射影 |
| `minimum` | 最小値 | + 射影 |
| `maximum` | 最大値 | + 射影 |
| `quartiles` | 四分位数（Q1, Q2, Q3） | + 射影 |
| `percentile` | 指定割合でのパーセンタイル | + 射影 |
| `five_number_summary` | 五数要約 | + 射影 |
| `weighted_median` | 重み付き中央値 | + 射影 |
| `weighted_percentile` | 重み付きパーセンタイル | + 射影 |

### 4. 分布の形状 (`shape_of_distribution.hpp`)

分布の形状を特徴付ける統計量。

**関数:**

| 関数 | 説明 | オーバーロード |
| ---- | ---- | -------------- |
| `population_skewness` | 母歪度 | + 事前計算済み平均, + 射影 |
| `sample_skewness` | 標本歪度 | + 事前計算済み平均, + 射影 |
| `skewness` | 歪度（デフォルトは標本歪度） | + 事前計算済み平均, + 射影 |
| `population_kurtosis` | 母尖度 | + 事前計算済み平均, + 射影 |
| `sample_kurtosis` | 標本尖度 | + 事前計算済み平均, + 射影 |
| `kurtosis` | 尖度（デフォルトは標本尖度） | + 事前計算済み平均, + 射影 |

### 5. 相関・共分散 (`correlation_covariance.hpp`)

2変数間の関係を測定する関数群。

**関数:**

| 関数 | 説明 | オーバーロード |
| ---- | ---- | -------------- |
| `population_covariance` | 母共分散 | + 事前計算済み平均, + 射影 |
| `sample_covariance` | 標本共分散 | + 事前計算済み平均, + 射影 |
| `covariance` | 共分散（デフォルトは標本共分散） | + 事前計算済み平均, + 射影 |
| `pearson_correlation` | Pearson 相関係数 | + 事前計算済み平均, + 射影 |
| `spearman_correlation` | Spearman 順位相関係数 | + 射影 |
| `kendall_tau` | Kendall の順位相関係数 | + 射影 |
| `weighted_covariance` | 重み付き共分散 | + 射影 |

### 6. 度数分布 (`frequency_distribution.hpp`)

度数分布とヒストグラムの作成。

**構造体:**

- `frequency_entry<T>` — フィールド: `value`, `count`, `relative_frequency`, `cumulative_count`, `cumulative_relative_frequency`
- `frequency_table_result<T>` — フィールド: `entries`, `total_count`

**関数:**

| 関数 | 説明 | オーバーロード |
| ---- | ---- | -------------- |
| `frequency_table` | 度数分布表 | + 射影 |
| `frequency_count` | 度数カウントマップ | + 射影 |
| `relative_frequency` | 相対度数マップ | + 射影 |
| `cumulative_frequency` | 累積度数 | + 射影 |
| `cumulative_relative_frequency` | 累積相対度数 | + 射影 |

### 7. 特殊関数 (`special_functions.hpp`)

統計計算で使用される特殊関数。

**定数:**

- `pi`, `sqrt_2`, `sqrt_2_pi`, `log_sqrt_2_pi`

**関数:**

| 関数 | 説明 |
| ---- | ---- |
| `lgamma` | 対数ガンマ関数 |
| `tgamma` | ガンマ関数 |
| `beta` | ベータ関数 |
| `lbeta` | 対数ベータ関数 |
| `betainc` | 正則化不完全ベータ関数 |
| `betaincinv` | 正則化不完全ベータ関数の逆関数 |
| `erf` | 誤差関数 |
| `erfc` | 相補誤差関数 |
| `norm_cdf` | 標準正規分布 CDF |
| `norm_quantile` | 標準正規分布分位点（逆 CDF） |
| `gammainc_lower` | 正則化下側不完全ガンマ関数 |
| `gammainc_upper` | 正則化上側不完全ガンマ関数 |
| `gammainc_lower_inv` | 正則化下側不完全ガンマ関数の逆関数 |

### 8. 乱数生成 (`random_engine.hpp`)

乱数生成エンジン。

**型エイリアス:**

- `default_random_engine` = `std::mt19937_64`

**型特性:**

- `is_random_engine<T>` — 乱数エンジン判定の型特性
- `is_random_engine_v<T>` — 変数テンプレートショートカット

**関数:**

| 関数 | 説明 |
| ---- | ---- |
| `get_random_engine` | スレッドローカルなデフォルト乱数エンジンを取得 |
| `set_seed` | 乱数シードを設定 |
| `randomize_seed` | ハードウェアエントロピーからシードをランダム化 |

### 9. 連続分布 (`continuous_distributions.hpp`)

連続確率分布の PDF、CDF、分位点関数、乱数生成。

各分布に `_pdf`, `_cdf`, `_quantile`, `_rand` 関数が提供されています（スチューデント化された範囲分布は CDF と分位点のみ）。

**関数:**

| 分布 | pdf | cdf | quantile | rand |
| ---- | --- | --- | -------- | ---- |
| 一様分布 | `uniform_pdf` | `uniform_cdf` | `uniform_quantile` | `uniform_rand` |
| 正規分布 | `normal_pdf` | `normal_cdf` | `normal_quantile` | `normal_rand` |
| 指数分布 | `exponential_pdf` | `exponential_cdf` | `exponential_quantile` | `exponential_rand` |
| ガンマ分布 | `gamma_pdf` | `gamma_cdf` | `gamma_quantile` | `gamma_rand` |
| ベータ分布 | `beta_pdf` | `beta_cdf` | `beta_quantile` | `beta_rand` |
| カイ二乗分布 | `chisq_pdf` | `chisq_cdf` | `chisq_quantile` | `chisq_rand` |
| t分布 | `t_pdf` | `t_cdf` | `t_quantile` | `t_rand` |
| F分布 | `f_pdf` | `f_cdf` | `f_quantile` | `f_rand` |
| 対数正規分布 | `lognormal_pdf` | `lognormal_cdf` | `lognormal_quantile` | `lognormal_rand` |
| ワイブル分布 | `weibull_pdf` | `weibull_cdf` | `weibull_quantile` | `weibull_rand` |
| Studentized range | — | `studentized_range_cdf` | `studentized_range_quantile` | — |

### 10. 離散分布 (`discrete_distributions.hpp`)

離散確率分布の PMF、CDF、分位点関数、乱数生成。

各分布に `_pmf`, `_cdf`, `_quantile`, `_rand` 関数が提供されています。

**ユーティリティ関数:**

| 関数 | 説明 |
| ---- | ---- |
| `log_factorial` | 階乗の対数 |
| `log_binomial_coef` | 二項係数の対数 |
| `binomial_coef` | 二項係数 |

**分布関数:**

| 分布 | pmf | cdf | quantile | rand |
| ---- | --- | --- | -------- | ---- |
| 二項分布 | `binomial_pmf` | `binomial_cdf` | `binomial_quantile` | `binomial_rand` |
| ポアソン分布 | `poisson_pmf` | `poisson_cdf` | `poisson_quantile` | `poisson_rand` |
| 幾何分布 | `geometric_pmf` | `geometric_cdf` | `geometric_quantile` | `geometric_rand` |
| 超幾何分布 | `hypergeom_pmf` | `hypergeom_cdf` | `hypergeom_quantile` | `hypergeom_rand` |
| 負の二項分布 | `nbinom_pmf` | `nbinom_cdf` | `nbinom_quantile` | `nbinom_rand` |
| ベルヌーイ分布 | `bernoulli_pmf` | `bernoulli_cdf` | `bernoulli_quantile` | `bernoulli_rand` |
| 離散一様分布 | `discrete_uniform_pmf` | `discrete_uniform_cdf` | `discrete_uniform_quantile` | `discrete_uniform_rand` |

### 11. 推定 (`estimation.hpp`)

統計的推定（信頼区間、サンプルサイズ計算）。

**構造体:**

- `confidence_interval` — フィールド: `lower`, `upper`, `point_estimate`, `confidence_level`

**関数:**

| 関数 | 説明 | オーバーロード |
| ---- | ---- | -------------- |
| `standard_error` | 平均の標準誤差 | + 事前計算済み標準偏差, + 射影 |
| `ci_mean` | 平均の信頼区間（t分布） | 2 オーバーロード |
| `ci_mean_z` | 平均の信頼区間（z、既知の分散） | |
| `ci_proportion` | 比率の信頼区間（Wald 法） | |
| `ci_proportion_wilson` | 比率の信頼区間（Wilson 法） | |
| `ci_variance` | 分散の信頼区間（カイ二乗ベース） | |
| `ci_mean_diff` | 平均の差の信頼区間 | |
| `ci_mean_diff_welch` | 平均の差の信頼区間（Welch 法） | |
| `ci_mean_diff_pooled` | 平均の差の信頼区間（プール法） | |
| `ci_proportion_diff` | 比率の差の信頼区間 | |
| `margin_of_error_mean` | 平均の誤差限界 | 2 オーバーロード |
| `margin_of_error_proportion` | 比率の誤差限界 | |
| `margin_of_error_proportion_worst_case` | 最悪ケースの誤差限界 | |
| `sample_size_for_moe_proportion` | 比率のサンプルサイズ計算 | |
| `sample_size_for_moe_mean` | 平均のサンプルサイズ計算 | |

### 12. パラメトリック検定 (`parametric_tests.hpp`)

パラメトリック仮説検定。

**列挙型:**

- `alternative_hypothesis` — 値: `two_sided`, `less`, `greater`

**構造体:**

- `test_result` — フィールド: `statistic`, `p_value`, `df`, `alternative`, `df2`

**関数:**

| 関数 | 説明 |
| ---- | ---- |
| `z_test` | 1標本 z 検定（既知の分散） |
| `z_test_proportion` | 1標本比率 z 検定 |
| `z_test_proportion_two_sample` | 2標本比率 z 検定 |
| `t_test` | 1標本 t 検定 |
| `t_test_two_sample` | 2標本 t 検定（プール分散） |
| `t_test_welch` | 2標本 t 検定（Welch 法） |
| `t_test_paired` | 対応のある t 検定 |
| `chisq_test_gof` | カイ二乗適合度検定 |
| `chisq_test_gof_uniform` | カイ二乗適合度検定（一様期待度数） |
| `chisq_test_independence` | カイ二乗独立性検定 |
| `f_test` | F 検定（等分散性の検定） |
| `bonferroni_correction` | Bonferroni p 値補正 |
| `benjamini_hochberg_correction` | Benjamini-Hochberg FDR 補正 |
| `holm_correction` | Holm ステップダウン補正 |

### 13. ノンパラメトリック検定 (`nonparametric_tests.hpp`)

ノンパラメトリック仮説検定。

**関数:**

| 関数 | 説明 |
| ---- | ---- |
| `compute_ranks_with_ties` | タイ処理付き順位計算 |
| `compute_tie_groups` | タイグループ情報の計算 |
| `shapiro_wilk_test` | Shapiro-Wilk 正規性検定 |
| `ks_test_normal` | Kolmogorov-Smirnov 正規性検定 |
| `levene_test` | Levene 検定（等分散性） |
| `bartlett_test` | Bartlett 検定（等分散性） |
| `wilcoxon_signed_rank_test` | Wilcoxon 符号順位検定 |
| `mann_whitney_u_test` | Mann-Whitney U 検定 |
| `kruskal_wallis_test` | Kruskal-Wallis 検定 |
| `fisher_exact_test` | Fisher の正確検定（2x2 分割表） |

### 14. 効果量 (`effect_size.hpp`)

効果量の計算。

**列挙型:**

- `effect_size_magnitude` — 値: `negligible`, `small`, `medium`, `large`

**関数:**

| 関数 | 説明 |
| ---- | ---- |
| `cohens_d` | Cohen's d（1標本、3 オーバーロード） |
| `cohens_d_two_sample` | Cohen's d（2標本、プール標準偏差） |
| `hedges_correction_factor` | Hedges' g バイアス補正係数 |
| `hedges_g` | Hedges' g（1標本、バイアス補正済み） |
| `hedges_g_two_sample` | Hedges' g（2標本、バイアス補正済み） |
| `glass_delta` | Glass's delta（対照群の標準偏差） |
| `t_to_r` | t 値から相関への変換 |
| `d_to_r` | Cohen's d から相関への変換 |
| `r_to_d` | 相関から Cohen's d への変換 |
| `eta_squared` | F 検定からのイータ二乗 |
| `partial_eta_squared` | 偏イータ二乗 |
| `omega_squared` | オメガ二乗（バイアス小） |
| `cohens_h` | Cohen's h（比率の効果量） |
| `odds_ratio` | オッズ比（2x2 分割表） |
| `risk_ratio` | リスク比（2x2 分割表） |
| `interpret_cohens_d` | Cohen's d の大きさの解釈 |
| `interpret_correlation` | 相関の大きさの解釈 |
| `interpret_eta_squared` | イータ二乗の大きさの解釈 |

### 15. リサンプリング (`resampling.hpp`)

リサンプリング法。

**構造体:**

- `bootstrap_result` — フィールド: `estimate`, `standard_error`, `ci_lower`, `ci_upper`, `bias`, `replicates`
- `permutation_result` — フィールド: `observed_statistic`, `p_value`, `n_permutations`, `permutation_distribution`

**関数:**

| 関数 | 説明 |
| ---- | ---- |
| `bootstrap_sample` | ブートストラップ標本の生成（2 オーバーロード） |
| `bootstrap` | カスタム統計量によるブートストラップ推定 |
| `bootstrap_mean` | 平均のブートストラップ |
| `bootstrap_median` | 中央値のブートストラップ |
| `bootstrap_stddev` | 標準偏差のブートストラップ |
| `bootstrap_bca` | BCa ブートストラップ信頼区間 |
| `permutation_test_two_sample` | 2標本置換検定 |
| `permutation_test_paired` | 対応のある置換検定 |
| `permutation_test_correlation` | 相関の置換検定 |

### 16. 検出力分析 (`power_analysis.hpp`)

検出力分析とサンプルサイズ計算。

**構造体:**

- `power_result` — フィールド: `power`, `sample_size`, `effect_size`, `alpha`

**関数（各関数は文字列版と `alternative_hypothesis` 列挙型版のオーバーロードあり）:**

| 関数 | 説明 |
| ---- | ---- |
| `power_t_test_one_sample` | 1標本 t 検定の検出力 |
| `sample_size_t_test_one_sample` | 1標本 t 検定のサンプルサイズ |
| `power_t_test_two_sample` | 2標本 t 検定の検出力 |
| `sample_size_t_test_two_sample` | 2標本 t 検定のサンプルサイズ |
| `power_prop_test` | 比率検定の検出力 |
| `sample_size_prop_test` | 比率検定のサンプルサイズ |
| `power_analysis_t_one_sample` | `power_result` を返す検出力分析 |
| `power_analysis_t_one_sample_n` | `power_result` を返すサンプルサイズ分析 |

### 17. 線形回帰 (`linear_regression.hpp`)

線形回帰分析。

**構造体:**

- `simple_regression_result` — フィールド: `intercept`, `slope`, `intercept_se`, `slope_se`, `intercept_t`, `slope_t`, `intercept_p`, `slope_p`, `r_squared`, `adj_r_squared`, `residual_se`, `f_statistic`, `f_p_value`, `df_regression`, `df_residual`, `ss_total`, `ss_regression`, `ss_residual`
- `multiple_regression_result` — フィールド: `coefficients`, `coefficient_se`, `t_statistics`, `p_values`, `r_squared`, `adj_r_squared`, `residual_se`, `f_statistic`, `f_p_value`, `df_regression`, `df_residual`, `ss_total`, `ss_regression`, `ss_residual`
- `prediction_interval` — フィールド: `prediction`, `lower`, `upper`, `se_prediction`
- `residual_diagnostics` — フィールド: `residuals`, `standardized_residuals`, `studentized_residuals`, `hat_values`, `cooks_distance`, `durbin_watson`

**関数:**

| 関数 | 説明 |
| ---- | ---- |
| `simple_linear_regression` | 単回帰分析 |
| `multiple_linear_regression` | 重回帰分析 |
| `predict` | 予測（2 オーバーロード: 単回帰、重回帰） |
| `prediction_interval_simple` | 単回帰の予測区間 |
| `confidence_interval_mean` | 単回帰の平均応答の信頼区間 |
| `compute_residual_diagnostics` | 残差診断（2 オーバーロード） |
| `compute_vif` | 分散膨張係数 |
| `correlation_matrix_determinant` | 相関行列の行列式 |
| `multicollinearity_score` | 多重共線性スコア |
| `r_squared` | 決定係数 |
| `adjusted_r_squared` | 自由度調整済み決定係数 |

### 18. 分散分析 (`anova.hpp`)

分散分析。

**構造体:**

- `anova_row` — フィールド: `source`, `ss`, `df`, `ms`, `f_statistic`, `p_value`
- `one_way_anova_result` — フィールド: `between`, `within`, `ss_total`, `df_total`, `n_groups`, `n_total`, `grand_mean`, `group_means`, `group_sizes`
- `two_way_anova_result` — フィールド: `factor_a`, `factor_b`, `interaction`, `error`, `ss_total`, `df_total`, `levels_a`, `levels_b`, `n_total`, `grand_mean`
- `posthoc_comparison` — フィールド: `group1`, `group2`, `mean_diff`, `se`, `statistic`, `p_value`, `lower`, `upper`, `significant`
- `posthoc_result` — フィールド: `method`, `comparisons`, `alpha`, `mse`, `df_error`
- `ancova_result` — フィールド: `ss_covariate`, `ss_treatment`, `ss_error`, `df_covariate`, `df_treatment`, `df_error`, `ms_covariate`, `ms_treatment`, `ms_error`, `f_covariate`, `f_treatment`, `p_covariate`, `p_treatment`, `adjusted_means`

**関数:**

| 関数 | 説明 |
| ---- | ---- |
| `one_way_anova` | 一元配置分散分析 |
| `two_way_anova` | 二元配置分散分析 |
| `tukey_hsd` | Tukey HSD 事後検定 |
| `bonferroni_posthoc` | Bonferroni 事後検定 |
| `dunnett_posthoc` | Dunnett 事後検定 |
| `scheffe_posthoc` | Scheffe 事後検定 |
| `one_way_ancova` | 一元配置共分散分析 |
| `eta_squared` | ANOVA 結果からのイータ二乗 |
| `partial_eta_squared_a` | 要因 A の偏イータ二乗 |
| `partial_eta_squared_b` | 要因 B の偏イータ二乗 |
| `partial_eta_squared_interaction` | 交互作用の偏イータ二乗 |
| `omega_squared` | ANOVA 結果からのオメガ二乗 |
| `cohens_f` | ANOVA 結果からの Cohen's f |

### 19. 一般化線形モデル (`glm.hpp`)

一般化線形モデル。

**列挙型:**

- `link_function` — 値: `identity`, `logit`, `probit`, `log`, `inverse`, `cloglog`
- `distribution_family` — 値: `gaussian`, `binomial`, `poisson`, `gamma_family`

**構造体:**

- `glm_result` — フィールド: `coefficients`, `coefficient_se`, `z_statistics`, `p_values`, `null_deviance`, `residual_deviance`, `df_null`, `df_residual`, `aic`, `bic`, `log_likelihood`, `iterations`, `converged`, `link`, `family`
- `glm_residuals` — フィールド: `response`, `pearson`, `deviance`, `working`

**関数:**

| 関数 | 説明 |
| ---- | ---- |
| `glm_fit` | GLM の当てはめ（IRLS アルゴリズム） |
| `logistic_regression` | ロジスティック回帰（binomial/logit） |
| `predict_probability` | GLM からの確率予測 |
| `odds_ratios` | ロジスティック回帰のオッズ比 |
| `odds_ratios_ci` | オッズ比と信頼区間 |
| `poisson_regression` | ポアソン回帰（poisson/log） |
| `predict_count` | ポアソンモデルからのカウント予測 |
| `incidence_rate_ratios` | ポアソン回帰の発生率比 |
| `compute_glm_residuals` | GLM 残差分析 |
| `overdispersion_test` | 過分散の検定 |
| `pseudo_r_squared_mcfadden` | McFadden 疑似決定係数 |
| `pseudo_r_squared_nagelkerke` | Nagelkerke 疑似決定係数 |

### 20. モデル選択 (`model_selection.hpp`)

モデル選択と正則化回帰。

**構造体:**

- `cv_result` — クロスバリデーション結果
- `regularized_regression_result` — 正則化回帰結果

**関数:**

| 関数 | 説明 |
| ---- | ---- |
| `aic` | 赤池情報量基準 |
| `aic_linear` | 線形回帰の AIC（2 オーバーロード） |
| `aicc` | 補正 AIC |
| `bic` | ベイズ情報量基準 |
| `bic_linear` | 線形回帰の BIC（2 オーバーロード） |
| `press_statistic` | PRESS 統計量 |
| `create_cv_folds` | クロスバリデーション分割の作成 |
| `cross_validate_linear` | 線形回帰のクロスバリデーション |
| `loocv_linear` | Leave-one-out クロスバリデーション |
| `ridge_regression` | Ridge 回帰 |
| `lasso_regression` | Lasso 回帰 |
| `elastic_net_regression` | Elastic Net 回帰 |
| `cv_ridge` | クロスバリデーション付き Ridge 回帰 |
| `cv_lasso` | クロスバリデーション付き Lasso 回帰 |
| `generate_lambda_grid` | 正則化パラメータグリッドの生成 |

### 21. 距離・類似度 (`distance_metrics.hpp`)

距離と類似度の計算。

**関数:**

| 関数 | 説明 | オーバーロード |
| ---- | ---- | -------------- |
| `euclidean_distance` | ユークリッド距離 | イテレータ, ベクトル |
| `manhattan_distance` | マンハッタン距離 | イテレータ, ベクトル |
| `cosine_similarity` | コサイン類似度 | イテレータ, ベクトル |
| `cosine_distance` | コサイン距離（1 - 類似度） | イテレータ, ベクトル |
| `mahalanobis_distance` | マハラノビス距離 | |
| `minkowski_distance` | ミンコフスキー距離 | イテレータ, ベクトル |
| `chebyshev_distance` | チェビシェフ距離 | イテレータ, ベクトル |

### 22. 数値ユーティリティ (`numerical_utils.hpp`)

数値計算のユーティリティ関数。

**定数:**

- `epsilon` — double の機械イプシロン
- `default_rel_tol` — デフォルト相対許容誤差 (1e-9)
- `default_abs_tol` — デフォルト絶対許容誤差 (1e-12)

**関数:**

| 関数 | 説明 |
| ---- | ---- |
| `approx_equal` | 浮動小数点数の近似等値判定 |
| `is_zero` | 値がほぼゼロか判定 |
| `is_finite` | 値が有限か判定 |
| `all_finite` | 範囲内の全値が有限か判定 |
| `has_converged_abs` | 絶対収束判定 |
| `has_converged_rel` | 相対収束判定 |
| `has_converged` | 複合収束判定 |
| `log1p_safe` | 数値安定な log(1 + x) |
| `expm1_safe` | 数値安定な exp(x) - 1 |
| `clamp` | 値の範囲制限 |
| `in_range` | 範囲判定 |
| `relative_error` | 相対誤差 |
| `safe_divide` | 安全な除算（ゼロ除算回避） |
| `kahan_sum` | Kahan 加算（2 オーバーロード） |
| `approx_equal_range` | 範囲の近似等値判定 |

### 23. 多変量解析 (`multivariate.hpp`)

多変量解析関数。

**構造体:**

- `pca_result` — PCA 分析結果

**関数:**

| 関数 | 説明 |
| ---- | ---- |
| `covariance_matrix` | 共分散行列 |
| `correlation_matrix` | 相関行列 |
| `standardize` | 標準化（z スコア） |
| `min_max_scale` | 最小最大スケーリング |
| `power_iteration` | べき乗法（固有値計算） |
| `pca` | 主成分分析 |
| `pca_transform` | PCA 結果によるデータ変換 |

### 24. 時系列分析 (`time_series.hpp`)

時系列分析関数。

**関数:**

| 関数 | 説明 |
| ---- | ---- |
| `autocorrelation` | 指定ラグの自己相関 |
| `acf` | 自己相関関数（全ラグ） |
| `pacf` | 偏自己相関関数 |
| `mae` | 平均絶対誤差 |
| `mse` | 平均二乗誤差 |
| `rmse` | 平均二乗誤差の平方根 |
| `mape` | 平均絶対パーセント誤差 |
| `moving_average` | 単純移動平均 |
| `exponential_moving_average` | 指数移動平均 |
| `diff` | 一次差分 |
| `seasonal_diff` | 季節差分 |
| `lag` | ラグ演算子 |

### 25. カテゴリカルデータ分析 (`categorical.hpp`)

カテゴリカルデータ分析。

**構造体:**

- `contingency_table_result` — 周辺度数付き分割表
- `odds_ratio_result` — 信頼区間付きオッズ比
- `relative_risk_result` — 信頼区間付き相対リスク
- `risk_difference_result` — 信頼区間付きリスク差

**関数:**

| 関数 | 説明 |
| ---- | ---- |
| `contingency_table` | 分割表の作成 |
| `odds_ratio` | オッズ比（分割表または 2x2 値） |
| `relative_risk` | 相対リスク（分割表または 2x2 値） |
| `risk_difference` | リスク差（分割表または 2x2 値） |
| `number_needed_to_treat` | 治療必要数 |

### 26. 生存時間解析 (`survival.hpp`)

生存時間解析関数。

**構造体:**

- `kaplan_meier_result` — Kaplan-Meier 推定結果
- `logrank_result` — ログランク検定結果
- `hazard_rate_result` — ハザード率推定結果

**関数:**

| 関数 | 説明 |
| ---- | ---- |
| `kaplan_meier` | Kaplan-Meier 生存率推定 |
| `logrank_test` | ログランク検定 |
| `median_survival_time` | 生存時間中央値 |
| `nelson_aalen` | Nelson-Aalen 累積ハザード推定 |

### 27. 頑健統計 (`robust.hpp`)

頑健な統計手法。

**構造体:**

- `outlier_detection_result` — 外れ値検出結果

**関数:**

| 関数 | 説明 |
| ---- | ---- |
| `mad` | 中央絶対偏差 |
| `mad_scaled` | スケーリング MAD（一致推定量） |
| `detect_outliers_iqr` | IQR 法による外れ値検出 |
| `detect_outliers_zscore` | z スコアによる外れ値検出 |
| `detect_outliers_modified_zscore` | 修正 z スコアによる外れ値検出 |
| `winsorize` | ウィンソライズ |
| `cooks_distance` | Cook の距離 |
| `dffits` | DFFITS 影響度 |
| `hodges_lehmann` | Hodges-Lehmann 推定量 |
| `biweight_midvariance` | バイウェイトミッドバリアンス |

### 28. クラスタリング (`clustering.hpp`)

クラスタリングアルゴリズム。

**列挙型:**

- `linkage_type` — 値: `single`, `complete`, `average`, `ward`

**構造体:**

- `kmeans_result` — フィールド: `labels`, `centroids`, `inertia`, `n_iter`
- `dendrogram_node` — フィールド: `left`, `right`, `distance`, `count`

**関数:**

| 関数 | 説明 |
| ---- | ---- |
| `euclidean_distance` | ユークリッド距離（ベクトル） |
| `manhattan_distance` | マンハッタン距離（ベクトル） |
| `kmeans_plusplus_init` | K-means++ 初期化 |
| `kmeans` | K-means クラスタリング |
| `hierarchical_clustering` | 階層的クラスタリング |
| `cut_dendrogram` | デンドログラムを k クラスタに分割 |
| `silhouette_score` | シルエットスコア |

### 29. データラングリング (`data_wrangling.hpp`)

データ変換と前処理。

**定数:**

- `NA` — NaN センチネル値

**構造体:**

- `group_result` — グループ化結果
- `aggregation_result` — 集約結果
- `label_encoding_result` — ラベルエンコーディング結果
- `validation_result` — データ検証結果

**関数:**

| 関数 | 説明 |
| ---- | ---- |
| `is_na` | 値が NA/NaN か判定 |
| `dropna` | NaN 値を除去（2 オーバーロード） |
| `fillna` | NaN を定数で補完 |
| `fillna_mean` | NaN を平均で補完 |
| `fillna_median` | NaN を中央値で補完 |
| `fillna_ffill` | NaN を前方補完 |
| `fillna_bfill` | NaN を後方補完 |
| `fillna_interpolate` | NaN を補間 |
| `filter` | 述語によるフィルタリング |
| `filter_rows` | 行列の行フィルタリング |
| `filter_range` | 値範囲によるフィルタリング |
| `log_transform` | 対数変換 |
| `log1p_transform` | log(1+x) 変換 |
| `sqrt_transform` | 平方根変換 |
| `boxcox_transform` | Box-Cox 変換 |
| `rank_transform` | 順位変換 |
| `group_by` | キー関数によるグループ化 |
| `group_mean` | グループ別平均 |
| `group_sum` | グループ別合計 |
| `group_count` | グループ別カウント |
| `sort_values` | 値のソート |
| `argsort` | データをソートするインデックス |
| `sample_with_replacement` | 復元抽出 |
| `sample_without_replacement` | 非復元抽出 |
| `stratified_sample` | 層化抽出 |
| `drop_duplicates` | 重複値の除去 |
| `value_counts` | 一意な値のカウント |
| `get_duplicates` | 重複値の取得 |
| `rolling_mean` | 移動平均 |
| `rolling_std` | 移動標準偏差 |
| `rolling_min` | 移動最小値 |
| `rolling_max` | 移動最大値 |
| `rolling_sum` | 移動合計 |
| `label_encode` | ラベルエンコーディング |
| `one_hot_encode` | ワンホットエンコーディング |
| `bin_equal_width` | 等幅ビニング |
| `bin_equal_freq` | 等頻度ビニング |
| `validate_data` | データ検証 |
| `validate_range` | 範囲検証 |

### 30. 欠損データ (`missing_data.hpp`)

欠損データの高度な処理。

**列挙型:**

- `missing_mechanism` — 値: `mcar`, `mar`, `mnar`, `unknown`

**構造体:**

- `mcar_test_result` — MCAR 検定結果
- `missing_pattern_info` — 欠損パターン情報
- `multiple_imputation_result` — 多重代入法結果
- `sensitivity_analysis_result` — 感度分析結果
- `tipping_point_result` — ティッピングポイント分析結果
- `complete_case_result` — 完全ケース分析結果

**関数:**

| 関数 | 説明 |
| ---- | ---- |
| `analyze_missing_patterns` | 欠損データパターンの分析 |
| `create_missing_indicator` | 欠損インディケータ行列の作成 |
| `test_mcar_simple` | 簡易 MCAR 検定 |
| `diagnose_missing_mechanism` | 欠損データメカニズムの診断 |
| `impute_conditional_mean` | 条件付き平均代入 |
| `multiple_imputation_pmm` | 多重代入法（PMM） |
| `multiple_imputation_bootstrap` | 多重代入法（ブートストラップ） |
| `sensitivity_analysis_pattern_mixture` | パターン混合感度分析 |
| `sensitivity_analysis_selection_model` | 選択モデル感度分析 |
| `find_tipping_point` | ティッピングポイント分析 |
| `extract_complete_cases` | 完全ケースの抽出 |
| `correlation_matrix_pairwise` | ペアワイズ完全相関行列 |

### 31. アンブレラヘッダー (`statcpp.hpp`)

全モジュールをインクルードする便利なヘッダー。追加の関数定義はありません。

```cpp
#include "statcpp/statcpp.hpp"  // すべてをインクルード
```

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
