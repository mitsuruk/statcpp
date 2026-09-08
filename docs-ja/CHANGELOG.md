# 変更履歴

statcpp ライブラリの変更履歴を記録します.

このプロジェクトは [Semantic Versioning](https://semver.org/) に従います.

## [0.4.0] - 2026-09-08

ライブラリ全体を R 4.4.2 と照合して発見した数値の是正. 0.3.0 の公開シグネチャは
すべてそのままコンパイルできる(API の変更は新規関数 1 つと末尾のデフォルト引数のみ).
ただし **`lilliefors_test()` の挙動は大きく変わる**ため, 下流の結果は変化する.
詳細は末尾の「アップグレード時の注意」を参照.

### Fixed (修正)

- **`nonparametric_tests.hpp` — `lilliefors_test()` / `ks_test_normal()`**: p 値を Dallal & Wilkinson (1986) の解析的近似に是正. 従来の `p = 2·exp(-2·d_adj²)` は裾用の式を全域に適用しており, `nortest::lillie.test` と全域で乖離していた. 正規に近い標本で R が 0.729 を返すところ 0.0295 を返し, **α = 0.05 で正規性を誤って棄却**していた. 新しい式は p ≤ 0.05 の範囲で R と完全一致(723/723 標本)し, 2,400 標本において α = 0.05 / 0.01 の棄却判定の不一致は 0 件. p > 0.10 は公表された適用範囲外であり, 正規性と矛盾しないことを示すのみ. D 統計量は不変.
- **`special_functions.hpp` — `norm_cdf()`**: `0.5·(1 + erf(x/√2))` から `0.5·erfc(-x/√2)` に変更. 従来の形は左裾で桁落ちし, x = -8.25 で既に 40% ずれ, x = -8.327 以降は厳密に 0 を返していた(真の値はまだ十分表現可能な範囲にある). 新しい形は x = -37.5 まで R と相対誤差 1.9e-13 で一致し, 中央域も 2.2e-14 から 1.7e-15 に改善. `normal_cdf()` と `lognormal_cdf()` も同時に是正される.
- **`parametric_tests.hpp`, `nonparametric_tests.hpp`, `glm.hpp`, `power_analysis.hpp` — 上側 p 値**: p 値を `1 - norm_cdf(z)` の形で求めていた 19 箇所を `norm_sf(z)` に変更. 従来は `|z| ≥ 8.30` で厳密に 0 を返していた. ポアソン回帰の切片 p 値は 0 から 1.2688719e-117 になり, R の 1.2688719e-117 と一致. `z_test()`, `z_test_proportion()`, `z_test_proportion_two_sample()`, `mann_whitney_u_test()`, `wilcoxon_signed_rank_test()`, GLM の Wald p 値, 検出力関数が対象.

### Added (追加)

- **`special_functions.hpp` — `norm_sf()`**: 標準正規分布の上側確率 `P(Z > x)` を `erfc` 経由で計算. 上側確率が必要な場面では `1 - norm_cdf(x)` ではなく本関数を使用すること.
- **`model_selection.hpp` — `cross_validate_linear()` / `cv_ridge()` / `cv_lasso()`**: 末尾に `bool shuffle = true` を追加し `create_cv_folds()` へ委譲. `false` を渡すと連続ブロックの決定的な fold となり, 交差検証が再現可能になる. 既定値は 0.3.0 と同じ挙動.

### Changed (変更)

- **`testWithR/`**: 単独プログラム `verify_vs_r`(57 関数を対象に 167 個の期待値を手作業で転記)を廃止し, 期待値を R から生成する Google Test 群に置き換え. **R と照合可能な公開関数 321 個すべて**を 164 テストで網羅し, 48,598 個の数値を突き合わせる. R は生成時に別プロセスとして実行するだけでリンクしないため, statcpp とテストバイナリは MIT ライセンスのまま.
- **`testWithR/VERIFIED_FUNCTIONS.md`, `testWithR/NON_VERIFIABLE_FUNCTIONS.md`**: `R_VERIFICATION_INVENTORY.ja.md`(全公開関数の分類)と `VERIFICATION_CHECKLIST.ja.md`(関数単位の進捗)に置き換え.

### Documentation (ドキュメント)

- `README.md`, `README.ja.md`, 両 `API_REFERENCE.md` の公開関数数を是正: ユニーク名 386 個, オーバーロードを含めて 538 個. 従来の 524 という数値は, それが記載されたコミットを含めどのコミットのコードとも一致しなかった.
- テスト数を 793 から, 単体テスト 857 件 + 照合テスト 164 件に是正.
- 両 API リファレンスに `norm_sf` を追加し, `betainc_impl` / `lgamma_impl` を公開インターフェースとして意図されていない実装補助関数として明記.
- 実装されていない Cauchy 分布への言及を削除(英語版 API リファレンスのみ).
- `testWithR/METHODOLOGY.md` を全面改訂. 分位数の type, `mad()` と平均絶対偏差の別物問題, `fivenum()` と type 7 分位数, 重み付き分散の意味論, `odds_ratios()` の切片の扱いなど, 実測した R との定義差 22 項目を記録.

### アップグレード時の注意

- **ソース互換性**: 削除・改名は一切なし. 既知の下流プロジェクトが参照する 302 個の `statcpp::` シンボルはすべて存在する.
- **結果の互換性**: `lilliefors_test()` は多くの入力で結論が変わる. 3,000 標本のうち α = 0.05 で 61.6% の棄却判定が反転し, そのすべてが 0.3.0 側の誤った棄却だった. 旧 p 値を固定していたゴールデンファイルや照合の許容誤差は再生成が必要.
- それ以外の変化は, 0.3.0 が厳密に 0 を返していた遠方の裾に限られる. `power_t_test_one_sample()`, `power_t_test_two_sample()`, `power_prop_test()` の変化は最大 2e-15, `sample_size_*` 5 関数は 3,653 ケースすべてで同一の整数を返した.

## [0.3.0] - 2026-07-09

全モジュールの計算手法レビューに基づく正確性・境界安全性の修正. 公開シグネチャは
すべて不変だが, 一部の関数は是正された値を返すようになる.

### Fixed (修正)

- **`linear_regression.hpp` — `compute_residual_diagnostics()`**: Cook's 距離の分母を英語版ヘッダで `(1-h)` から `(1-h)²` に是正(日本語版は 0.2.0 で修正済み. `include/` と `include-ja/` の同期ずれを解消). 高レバレッジ点で最大約6倍の過小評価だった.
- **`discrete_distributions.hpp` — `poisson_quantile()` / `geometric_quantile()` / `nbinom_quantile()`**: `prob == 1.0` のガードを追加. 従来は `+inf` を `uint64` へキャスト(未定義動作). 無限台のため `std::numeric_limits<uint64_t>::max()` を返す.
- **`continuous_distributions.hpp` — `beta_rand()`**: 極小 shape(例 α=β=0.001)で両 gamma 変量が 0 にアンダーフローした場合に再抽選. 従来は無警告で NaN を返していた.
- **`data_wrangling.hpp` — `rolling_mean()` / `rolling_sum()`**: NaN は実際に含む窓のみを NaN にする. 従来は差分更新の和が一度 NaN になると以降すべて NaN だった. `rolling_std/min/max` と整合.
- **`glm.hpp` — `glm_fit()` の係数 SE**: 係数共分散に分散 φ を適用(`φ·(XᵀWX)⁻¹`). φ は binomial/poisson で 1 固定, gaussian/gamma は Pearson 統計量 / 残差df で推定(R の `summary.glm` と同じ). gaussian の SE が OLS と一致し, z 統計量・p 値も連動して是正.
- **`glm.hpp` — Poisson の null 対数尤度**: 欠落していた `-lgamma(y+1)` 項を追加. McFadden 擬似 R² が null とモデルで整合.
- **`glm.hpp` — gamma の対数尤度**: 厳密な gamma(shape=ν, mean=μ) 密度 `ν·log(ν/μ) − logΓ(ν) + (ν−1)·log(y) − ν·y/μ` に是正(従来は `ν·log ν` 項が欠落し `log y` の係数が `2ν−2`). gamma の AIC/BIC に影響.
- **`nonparametric_tests.hpp` — `kruskal_wallis_test()`**: 全同値入力で tie 補正の除数が 0 となり `-inf`/NaN を返していた. H=0, p=1 を返すよう是正.
- **`nonparametric_tests.hpp` — `shapiro_wilk_test()`**: W ≥ 1(最も正規的, 例: 完全等間隔データ)で p 値が約 0.001(正規性を強く棄却)だったのを約 1 に是正.

## [0.2.0] - 2026-03-13

### Added (追加)

- **`nonparametric_tests.hpp` — `mann_whitney_u_test()`**: 連続性補正パラメータ `correct=true` を追加(R の `wilcox.test` と同等).
- **`basic_statistics.hpp`, `dispersion_spread.hpp`, `order_statistics.hpp`**: 明示的な `WeightIterator` パラメータを持つ新しい重み付き API オーバーロードを追加. 旧 3 引数オーバーロードは `[[deprecated]]` に.
- **`basic_statistics.hpp`, `order_statistics.hpp`**: ランダムアクセスイテレータ要件の `static_assert` を追加.
- **`dispersion_spread.hpp` — `weighted_variance()` / `weighted_stddev()`**: 信頼性重み(reliability weights)セマンティクスの新オーバーロードを追加.

### Fixed (修正)

- **`robust.hpp` — `biweight_midvariance()`**: 分母の重み関数を `(1-u²)²` から `(1-u²)` に修正.
- **`order_statistics.hpp` — `weighted_median()` / `weighted_percentile()`**: 累積重みが境界に達した際に重み 0 の要素をスキップして次の正重み要素を探すように修正.
- **`special_functions.hpp` — `erf()` / `erfc()`**: カスタム近似を `std::erf()` / `std::erfc()` に置換. 完全精度に改善.
- **`discrete_distributions.hpp` — `discrete_uniform_quantile()`**: 計算を `floor(p * range)` から `ceil(p * range - 1)` に修正.
- **`linear_regression.hpp` — `cook_distance()`**: 分母を `(1-h)` から `(1-h)²` に修正(単回帰・重回帰の両方).
- **`glm.hpp` — `glm_fit()`**: binomial/poisson で `y_mean` クリッピング前の値を保持し,ヌル逸脱度計算に使用.
- **`clustering.hpp` — `kmeans()`**: K-means++ 初期化で `total_dist=0` 時のフォールバックを追加. 空クラスタの最遠点再初期化を追加.
- **`power_analysis.hpp` — `power_prop_test()`**: 比率検定の検出力計算を 2 段階方式に修正.
- **`data_wrangling.hpp` — `rank_transform()`**: NaN を含むデータに対応. NaN 位置には NaN 順位を割り当て.
- **`basic_statistics.hpp` — `weighted_harmonic_mean()`**: ゼロ近傍値判定を `harmonic_mean` と統一.
- **`continuous_distributions.hpp` — `beta_pdf()` / `gamma_pdf()`**: 境界値処理を修正.
- **`missing_data.hpp`**: `m >= 2` の入力検証と行長一致検証を追加.
- **`basic_statistics.hpp` — `mean()`**: 内部累積を `double` に変更し整数オーバーフローを防止.
- **`glm.hpp` — `glm_fit()`**: Gaussian AIC/BIC で σ² を推定パラメータとしてカウントするよう修正.
- **`order_statistics.hpp` — `weighted_percentile()`**: 厳密比較を許容誤差付き比較に変更.
- **`resampling.hpp`**: `n_bootstrap < 2` バリデーションと BCa インデックスクランプを追加.

### Changed (変更)

- **`nonparametric_tests.hpp` — `ks_test_normal()`**: `lilliefors_test()` にリネーム. 旧名は `[[deprecated]]` エイリアスとして保持.
- **`missing_data.hpp` — `test_mcar_simple()`**: 「Little の MCAR 検定」を「MCAR 簡易検定(平均差ベース)」に緩和.
- **`dispersion_spread.hpp` — `weighted_variance()`**: 重みのセマンティクスを「信頼性重み(reliability weights)」として文書化.
- **ヘッダーガード**: 全ヘッダーを `#pragma once` に統一.
- **`model_selection.hpp`**: `detail::standardize_features()` と `detail::rescale_coefficients()` ヘルパーを抽出し,コード重複を削減.
- **`estimation.hpp` — `ci_mean_diff_pooled()`**: `ci_mean_diff()` に委譲するよう簡略化(同一ロジック).

### Tests (テスト)

- Google Test による 847 件の単体テスト(v0.1.0 時点で 758 件)
- R 4.4.2 に対する 167 件の数値検証チェック
- `test_distance_metrics.cpp` 追加(41 テスト)
- erf/erfc NIST 精度テスト追加(5 テスト)
- 重み付き分散/標準偏差テスト追加(8 テスト)

### Documentation (ドキュメント)

- サンプルコードの `q.Q1`/`q.Q3` を `q.q1`/`q.q3` に修正.
- 該当しない機能(ジャックナイフ,反復測定 ANOVA)を機能一覧から削除.
- `github.com/yourusername/statcpp` プレースホルダを `github.com/mitsuruk/statcpp` に置換.
- 動作確認環境に `macOS + GCC 15 (Homebrew)` を追記.
- `distance_metrics.hpp` のコメントを日本語に翻訳(JA 版).
