# 変更履歴

statcpp ライブラリの変更履歴を記録します.

このプロジェクトは [Semantic Versioning](https://semver.org/) に従います.

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
