# NaN(欠損値)の取り扱い方針

> **状態: v0.5.0 で実装済み(2026-10-07)**

statcpp の全公開関数が、入力に NaN を含むときにどう振る舞うかを定める。
v0.4.0 以前はこの方針が無く、関数ごとに挙動が異なっていた(第 6 節「v0.4.0 の問題」)。
v0.5.0 で全公開関数をこの方針に揃えた(第 9 節、第 10 節)。

---

## 1. 用語

- **NaN**: `std::numeric_limits<double>::quiet_NaN()`。statcpp では欠損値(NA)をこれで表す
  (`statcpp::NA` 定数と同じ)。NaN の種類(quiet / signaling、符号)は区別しない。
- **±Inf**: 欠損値では**ない**。通常の値として扱う(本方針の対象外)。
- **データ引数**: 観測値の列(イテレータ範囲、`std::vector`、行列)。重み付き統計量の重みも含む。
- **パラメータ引数**: 観測値以外の数値(`mu`、`sigma`、`alpha`、信頼水準、窓幅など)。
- **評価点**: 分布関数などで値を求める点(`normal_pdf(x, ...)` の `x`、`normal_quantile(p, ...)` の `p`)。

---

## 2. 原則

1. **R 準拠**: 各関数は、対応する R の関数を**既定の引数で**呼んだときの NA 処理に従う。

   | R の挙動 | statcpp の挙動 |
   | --- | --- |
   | `NA` / `NaN` を返す | NaN を返す |
   | NA を除去して計算する | NaN を除去して計算する |
   | エラーになる | `std::invalid_argument` を送出する |

2. **R に対応する関数が無い場合**は、第 4 節のクラス既定に従う。
3. **R と異なる扱いをする場合**は、第 5 節に理由とともに列挙し、`testWithR/METHODOLOGY.md` にも記録する。
4. **黙って壊れない**: どの関数も、NaN を含む入力で無限ループ・未定義動作・NaN の位置に依存する結果を
   起こしてはならない(R 準拠以前の最低条件)。

R 照合テストに NA を含むケースを加えることで、本方針への準拠を機械的に検証する。

---

## 3. R の既定の挙動(R 4.4.2 で実測)

入力 `c(3, 1, NA, 4, 5, 2)` などで実測した結果。statcpp の対応関数はこれに従う。

| 区分 | R の挙動 | R の関数の例 |
| --- | --- | --- |
| 記述統計 | NA を返す | `mean`、`var`、`sd`、`min`、`max`、`range`、`median`、`mad`、`cor`(pearson / spearman) |
| 記述統計 | エラー | `quantile`、`IQR` |
| 記述統計 | NA を除去して計算 | `fivenum` |
| 要素ごとの出力 | 該当要素だけ NA | `diff`、`log` など要素ごとの変換 |
| 累積型の出力 | NA 以降がすべて NA | `cumsum` |
| 並べ替え | NA を除去 | `sort` |
| 並べ替え | NA を末尾に置く | `order` |
| 推測統計 | NA を除去して計算(2 標本の対応ありは対ごとに除去) | `t.test`、`wilcox.test`、`shapiro.test`、`ks.test`、`var.test`、`cor.test`、`kruskal.test`、`aov`、`lm` |
| 推測統計 | エラー | `chisq.test`、`kmeans`、`prcomp` |
| 数学関数 | NaN を返す | `dnorm(NaN)`、`qnorm(NaN)`、`lgamma(NaN)` |
| パラメータ | NaN を返す(警告のみ) | `dnorm(0, 0, NaN)`、`rnorm(1, 0, NaN)`、`dpois(2, NaN)` |

関数ごとの対応(statcpp の関数 → R の関数と既定の挙動、方針での挙動)は、付録 [NAN_INVENTORY.md](NAN_INVENTORY.md) に載せる。

---

## 4. R に対応する関数が無い場合のクラス既定

| クラス | 対象 | 既定 |
| --- | --- | --- |
| C1 スカラー統計量 | 1 つの実数を返す記述統計・距離など | NaN を返す(R の記述統計の大多数と同じ) |
| C2 要素ごと・窓ごとの出力 | rolling 系、変換系、ラグ・差分 | NaN に依存する要素・窓だけ NaN にする。累積型(`exponential_moving_average` など)は R の `cumsum` と同じく、NaN 以降を NaN にする |
| C3 整数・カテゴリの出力 | ビン番号・クラス番号・度数 | `std::invalid_argument` を送出する(第 5 節) |
| C6 推測統計・モデル推定 | 構造体を返す検定・推定・モデル | NaN を除去して計算する(R の検定の大多数と同じ)。2 標本の対応ありの関数は対ごとに除去する |
| C7 数学関数 | 分布関数・特殊関数 | 評価点またはパラメータが NaN なら NaN を返す |

---

## 5. R と異なる扱い(決定済み)

| 対象 | R | statcpp | 理由 |
| --- | --- | --- | --- |
| 整数・列挙を返す関数(`bin_equal_width`、`bin_equal_freq`、`label_encode`、`one_hot_encode`、`value_counts`、`frequency_*`、離散分布の `*_quantile` / `*_rand`、`sample_size_*`、`interpret_*`、外れ値検出の `detect_outliers_*`) | NA を返す、または NA を除外する | `std::invalid_argument` を送出する | 戻り値の整数型・列挙型(外れ値検出では点ごとの外れ値フラグ)では NA を表せない。戻り値の型を変える API 変更は行わない |
| 範囲検査のあるパラメータ(信頼水準、`alpha`、分位点の `p`、トリム率、`lambda`、`tol`、窓幅、抽出比率など)が NaN | 関数により NA またはエラー | `std::invalid_argument` を送出する | 範囲外の値と同じく不正なパラメータとして扱う。分布関数のパラメータは対象外(R どおり NaN を返す) |
| `filter_range`、`validate_range` の境界が NaN | 対応なし | `std::invalid_argument` を送出する | 境界が NaN だと検証が素通りになる(現状は常に true を返す) |
| `fillna_ffill`、`fillna_bfill`、`fillna_interpolate` | `zoo::na.locf` / `zoo::na.approx` の既定は端の NA を削除して長さが縮む | 長さを保ち、補完できない端の NaN は NaN のまま残す | 入力と要素が対応するベクトルを返す仕様を保つ |
| `rank_transform`、`compute_ranks_with_ties` | `rank` は既定(`na.last = TRUE`)で NA に最大の順位を付ける | NaN を NaN のまま残す(`rank(na.last = "keep")` 相当) | NaN が普通の値として順位に混ざると、後続の計算で誤用しやすい |
| クラスタリング(`hierarchical_clustering`、`silhouette_score`) | `kmeans` はエラー、`dist()` 経由の関数は NA の座標を除いて計算する | `std::invalid_argument` を送出する | クラスタリング関数の扱いを `kmeans` に揃えて統一する |
| `two_way_anova` | 非平衡のデータでも計算する | NaN を除去したうえで、既存の平衡データの検査に従う(非平衡なら例外) | statcpp の `two_way_anova` は平衡データを前提としている |
| 有限だが不正なパラメータ(`sigma <= 0` など) | 警告付きで NaN を返す | 従来どおり `std::invalid_argument` を送出する | 既存の仕様。本方針は NaN の扱いだけを変える |

---

## 6. v0.4.0 の問題(v0.5.0 で修正済み)

全 391 関数(モジュール単位、同名の重複を含む)を実測した結果を付録 [NAN_INVENTORY.md](NAN_INVENTORY.md) に載せる。
NaN を含む入力を「途中」と「先頭」の 2 通りで与え、1 関数ずつ別プロセスで 10 秒の時間制限を付けて実行した。
通常ビルドに加えて AddressSanitizer / UndefinedBehaviorSanitizer 付きビルドでも実行し、R 4.4.2 の既定引数の結果と照合した。

| 判定 | 件数 | 意味 |
| --- | --- | --- |
| OK | 200 | 現状で本方針に準拠している |
| FIX | 153 | 本方針と異なる(多くは黙って誤った値を返す) |
| FIX-UB | 22 | 未定義動作(NaN の整数キャスト、NaN を含む比較での `std::sort`) |
| FIX-HANG | 16 | 無限ループ |

根本原因は次のパターンに集約される。

| 根本原因 | 代表的な関数 |
| --- | --- |
| 同値をまとめるループが NaN で進まない(`NaN == NaN` が偽) | `spearman_correlation`、`mann_whitney_u_test`、`wilcoxon_signed_rank_test`、`kruskal_wallis_test`、`kaplan_meier`、`nelson_aalen`(無限ループ) |
| NaN のパラメータを標準ライブラリの分布に渡している | `gamma_rand`、`poisson_rand` など 8 関数(無限ループ) |
| NaN を含む `std::sort` / `std::min_element` | `minimum`、`maximum`、`range`、`mad`、`weighted_median`、`holm_correction`、`sort_values` |
| NaN を整数にキャストしている | `percentile`、`trimmed_mean`、`bootstrap` 系、`bin_equal_width`、`discrete_uniform_quantile` |
| パラメータの NaN が `x <= 0` 形式の検査をすり抜ける | 信頼水準・`alpha` を取る推定・検定・検出力関数 |
| `std::map<double>` のキーに NaN が入る | `value_counts`、`group_*`、`frequency_*`、`label_encode` |
| NaN を除去していない(R は除去する) | `t_test`、`ci_mean` など推測統計の多く |

黙って誤った結論を返していた例:

| 関数 | 現状 |
| --- | --- |
| `permutation_test_two_sample` ほか 2 関数 | NaN が 1 つあると p = 0.002(NaN が無いときは p = 0.85)で、偽の有意になる |
| `chisq_test_independence` | NaN のセルがあると χ² = 0、p = 1 |
| `shapiro_wilk_test` | W = 1、p = 1(誤って「正規分布である」と結論する) |
| `lilliefors_test`、`ks_test_normal` | D = 0、p = 1 |
| GLM 系 | 係数が NaN なのに `converged = true` を返す |
| `detect_outliers_iqr` | NaN が 1 つあると全点を外れ値と判定する |
| `tukey_hsd` | p = 2.4e-14 |
| `percentile` 系 | NaN の位置によって変わる有限の値を返す |

なお、NaN とは無関係の不具合として、`fillna_median` が未ソートのまま `median`(ソート済みの入力が前提)を呼び、
誤った中央値で補完していることも判明した(`{9, NaN, 1, 5, 3}` の補完値が、正しくは 4 のところ 3 になる)。v0.5.0 で修正した。

---

## 7. 実装上の取り決め

- NaN の判定・除去・末尾への並べ替えは `statcpp::detail` の共通ヘルパ(`nan_utils.hpp`)に集約し、各関数に同じコードを散らさない。
- 例外メッセージは既存の形式 `"statcpp::<関数名>: <理由>"` に従う(例: `"statcpp::bin_equal_width: data contains NaN"`)。
- NaN を除去して計算する関数は、除去後のデータで既存の検査(データ数不足など)を行う。
- R と照合できる関数は R 照合テストに NA ケースを追加し、照合できない関数は単体テストでクラス既定を検証する。

---

## 8. 対象外

- ±Inf の扱い(通常の値として扱う現状を維持する)。ただし `moving_average` は、累積和で Inf − Inf が NaN になり
  以降の窓がすべて NaN になる不具合があったため、v0.5.0 で R の `stats::filter` と同じ窓ごとの値に修正した。
- 有限だが不正なパラメータの扱い(第 5 節のとおり従来どおり)。
- NaN の種類(signaling / quiet、符号、ペイロード)の区別。

---

## 9. 実装で確定した個別の判断

第 4 節・第 5 節だけでは決まらなかった関数は、次のとおり実装した。R に対応がある場合はその既定の挙動に従っている。

| 対象 | v0.5.0 の挙動 | 根拠 |
| --- | --- | --- |
| `sort_values` / `argsort` | `sort_values` は NaN を除く。`argsort` は NaN を元の順序のまま末尾に置く | R の `sort` / `order(na.last = TRUE)` |
| `argmin` / `argmax` | NaN を除いた中での位置を返す。すべて NaN なら例外 | R の `which.min` / `which.max` |
| `drop_duplicates` / `get_duplicates` | NaN 同士を同じ値とみなす | R の `unique` / `duplicated` |
| `group_by`、`group_mean`、`group_sum`、`group_count` | キーが NaN の行を除く | R の `split` / `tapply` は NA のグループを作らない |
| `rolling_min` / `rolling_max` | NaN を含む窓は位置によらず NaN | クラス既定 C2 |
| `clamp` | どの引数が NaN でも NaN を返す | R の `pmin(pmax(x, lo), hi)` |
| `euclidean_distance`、`manhattan_distance`、`minkowski_distance`(両ヘッダ) | NaN の座標を除き、和を n / n_used 倍に補正する。使える座標が無ければ NaN | R の `dist()` |
| `cosine_similarity` / `cosine_distance` | NaN を含む組を除く。使える組が無ければ NaN | `proxy::simil`(base R に対応なし) |
| `kendall_tau`、`spearman_correlation` | NaN を返す | R の `cor(method = ...)` の既定(`use = "everything"`) |
| `weighted_covariance`、`weighted_variance`、`weighted_stddev` | 例外 | R の `cov.wt` はエラー |
| `autocorrelation`、`acf`、`pacf` | 例外 | R の `acf` の既定 `na.action = na.fail` |
| `ci_mean`、`ci_mean_z`、`ci_variance`、`margin_of_error_mean`、`ci_mean_diff*` | NaN を除いて計算する | R の `t.test` と揃える(クラス既定 C6) |
| 回帰・GLM・交差検証(`simple_linear_regression`、`glm_fit` ほか) | NaN を含む行を除く(残差などの長さも除去後になる) | R の `lm` / `glm` の `na.omit` |
| `pseudo_r_squared_nagelkerke` | NaN の応答を除き、その数だけ n を減らす | `na.omit` で推定したモデルと整合させる |
| 正則化回帰、`cv_ridge`、`cv_lasso`、`generate_lambda_grid` | 例外。`lambda` グリッドの NaN も例外 | `glmnet` は NA を含む入力でエラー |
| `correlation_matrix` | 要素ごとに NaN を伝播し、対角は 1 | R の `cor` |
| `standardize` | 列の統計量を NaN 以外から求め、NaN のセルだけ NaN | R の `scale` |
| `power_iteration`、`pca`、`pca_transform`、クラスタリング | 例外 | R の `prcomp` / `kmeans` はエラー |
| `permutation_test_*` | NaN を除く。対応のある 2 関数は対ごとに除く | R の `t.test` / `cor.test` と同じ扱い。v0.4.0 は偽の有意(p = 1/(B+1))を返していた |
| `bootstrap` / `bootstrap_bca`(汎用) | データはそのまま統計量関数に渡す。信頼水準が NaN なら例外 | R の `boot` |
| `bootstrap_mean` / `bootstrap_median` / `bootstrap_stddev` | NaN を除く | クラス既定 C6 |
| 連続分布の `*_rand` | パラメータが NaN なら NaN を返す | R の `rnorm(1, 0, NaN)` など |
| `multiple_imputation_bootstrap` | ブートストラップ標本に観測値が無い列は元データの観測値から補完し、列全体が欠損なら NaN のまま | v0.4.0 は黙って 0 で補完していた |
| `mu0` などの検定の基準値 | NaN なら例外 | R の `t.test(mu = NA)` はエラー |
| 事後検定の `alpha` | NaN なら例外 | 第 5 節の範囲検査のあるパラメータ |

---

## 10. 検証

- **単体テスト**: 各モジュールのテストに「NaN Handling (v0.5.0)」の節を設け、修正前に失敗することを確認してから修正した。
  NaN ヘルパのテスト(`test_nan_utils.cpp`)を含め、`statcpp_tests` は全 974 件。
- **R 照合**: `testWithR/test_vs_r_nan.cpp` が、NA を含む入力に対する R 4.4.2 の既定の結果と 23 ケースを照合する
  (除去、伝播、要素ごと、距離の補正、行の除去、並べ替え)。同じテストを v0.4.0 のヘッダで実行すると 23 件すべてが
  失敗または停止する。
- **構成**: Debug、Release、RelWithDebInfo、日本語ヘッダ、libc++ hardening、AddressSanitizer / UndefinedBehaviorSanitizer の
  6 構成で全 1161 件が合格し、UBSan の報告は 0 件。
