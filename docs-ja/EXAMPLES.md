# サンプルコード

statcpp ライブラリの各機能を示すサンプルプログラムが `examples-ja/` ディレクトリに用意されています。

## サンプルプログラム一覧

### 基本統計量・散布度

| ファイル名 | 内容 |
|-----------|------|
| `example_basic_statistics.cpp` | 平均、中央値、最頻値などの基本統計量 |
| `example_dispersion_spread.cpp` | 分散、標準偏差、範囲などの散布度 |
| `example_order_statistics.cpp` | 四分位数、パーセンタイル、五数要約 |
| `example_shape_of_distribution.cpp` | 歪度、尖度などの分布の形状 |
| `example_frequency_distribution.cpp` | 度数分布、ヒストグラム |

### 相関・共分散

| ファイル名 | 内容 |
|-----------|------|
| `example_correlation_covariance.cpp` | 相関係数、共分散の計算 |

### 特殊関数

| ファイル名 | 内容 |
|-----------|------|
| `example_special_functions.cpp` | ガンマ関数、ベータ関数、誤差関数など |
| `example_numerical_utils.cpp` | 数値計算ユーティリティ（Kahan加算など） |

### 確率分布

| ファイル名 | 内容 |
|-----------|------|
| `example_random_engine.cpp` | 乱数生成エンジン |
| `example_continuous_distributions.cpp` | 正規分布、t分布、F分布などの連続分布 |
| `example_discrete_distributions.cpp` | 二項分布、ポアソン分布などの離散分布 |

### 統計的推定・検定

| ファイル名 | 内容 |
|-----------|------|
| `example_estimation.cpp` | 信頼区間の推定 |
| `example_parametric_tests.cpp` | t検定、z検定などのパラメトリック検定 |
| `example_nonparametric_tests.cpp` | Wilcoxon検定、Mann-Whitney検定などのノンパラメトリック検定 |
| `example_effect_size.cpp` | Cohen's d、相関比などの効果量 |
| `example_power_analysis.cpp` | 検出力分析、サンプルサイズ計算 |

### リサンプリング・回帰分析

| ファイル名 | 内容 |
|-----------|------|
| `example_resampling.cpp` | ブートストラップ、ジャックナイフ、置換検定 |
| `example_linear_regression.cpp` | 単回帰、重回帰分析 |
| `example_anova.cpp` | 分散分析（ANOVA） |
| `example_glm.cpp` | 一般化線形モデル（GLM） |

### 多変量解析・クラスタリング

| ファイル名 | 内容 |
|-----------|------|
| `example_multivariate.cpp` | 多変量解析 |
| `example_clustering.cpp` | k-means、階層的クラスタリング |
| `example_distance_metrics.cpp` | ユークリッド距離、マンハッタン距離などの距離メトリクス |

### 時系列・カテゴリカルデータ

| ファイル名 | 内容 |
|-----------|------|
| `example_time_series.cpp` | 時系列分析 |
| `example_categorical.cpp` | カテゴリカルデータ分析 |

### データ処理・頑健統計

| ファイル名 | 内容 |
|-----------|------|
| `example_data_wrangling.cpp` | データ変換、欠損値処理 |
| `example_missing_data.cpp` | 欠損値の高度な処理 |
| `example_robust.cpp` | 頑健統計（外れ値に強い統計量） |
| `example_survival.cpp` | 生存時間解析 |

### その他

| ファイル名 | 内容 |
|-----------|------|
| `example_model_selection.cpp` | モデル選択（AIC、BICなど） |

## サンプルプログラムのビルドと実行

### 方法1: シェルスクリプトを使用（macOS/Linux）

すべてのサンプルプログラムを一括でビルドして実行:

```bash
cd Examples-jp
./build.sh
```

このスクリプトは:
1. すべての `.cpp` ファイルをコンパイル
2. 実行ファイルを生成
3. 各プログラムを実行して出力を表示

### 方法2: 個別にコンパイル

特定のサンプルプログラムだけをビルド:

```bash
cd Examples-jp

# GCC を使用
g++ -std=c++17 -I../include example_basic_statistics.cpp -o example_basic_statistics

# Clang を使用
clang++ -std=c++17 -I../include example_basic_statistics.cpp -o example_basic_statistics

# 実行
./example_basic_statistics
```

### 方法3: CMake を使用

CMake を使ってすべてのサンプルをビルド:

```bash
cd Examples-jp
mkdir build
cd build
cmake ..
cmake --build .

# 実行
./example_basic_statistics
./example_dispersion_spread
# ...
```

## サンプルコードの読み方

各サンプルプログラムは以下の構成になっています:

1. **ヘッダーのインクルード**: 使用する機能のヘッダーファイルをインクルード
2. **データの準備**: サンプルデータの定義
3. **関数の呼び出し**: statcpp の関数を使用した計算
4. **結果の表示**: 計算結果の出力

### 基本的な例: `example_basic_statistics.cpp`

```cpp
#include <iostream>
#include <vector>
#include "statcpp/basic_statistics.hpp"

int main() {
    // 1. データの準備
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};

    // 2. 関数の呼び出し
    double avg = statcpp::mean(data.begin(), data.end());
    double total = statcpp::sum(data.begin(), data.end());
    std::size_t n = statcpp::count(data.begin(), data.end());

    // 3. 結果の表示
    std::cout << "データ数: " << n << std::endl;
    std::cout << "合計: " << total << std::endl;
    std::cout << "平均: " << avg << std::endl;

    return 0;
}
```

## カスタマイズ

サンプルプログラムは学習用に設計されています。自分のデータやユースケースに合わせて以下のようにカスタマイズできます:

### データの変更

```cpp
// サンプルデータを自分のデータに置き換え
std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
// ↓
std::vector<double> data = {/* あなたのデータ */};
```

### ファイルからのデータ読み込み

```cpp
#include <fstream>
#include <vector>

std::vector<double> read_data(const std::string& filename) {
    std::vector<double> data;
    std::ifstream file(filename);
    double value;
    while (file >> value) {
        data.push_back(value);
    }
    return data;
}

int main() {
    auto data = read_data("data.txt");
    double avg = statcpp::mean(data.begin(), data.end());
    // ...
}
```

### 複数の統計量を一度に計算

```cpp
#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"
#include "statcpp/order_statistics.hpp"

void analyze_data(const std::vector<double>& data) {
    double avg = statcpp::mean(data.begin(), data.end());
    double sd = statcpp::stddev(data.begin(), data.end());

    auto sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    double med = statcpp::median(sorted_data.begin(), sorted_data.end());

    std::cout << "平均: " << avg << std::endl;
    std::cout << "標準偏差: " << sd << std::endl;
    std::cout << "中央値: " << med << std::endl;
}
```

## トラブルシューティング

### コンパイルエラー: ヘッダーが見つからない

インクルードパスが正しく指定されているか確認してください:

```bash
g++ -std=c++17 -I/path/to/statcpp/include example.cpp -o example
```

### 実行時エラー: invalid_argument

多くの統計関数は空のデータや不正な引数で例外を投げます。データが適切か確認してください:

```cpp
std::vector<double> data = {/* データ */};
if (data.empty()) {
    std::cerr << "データが空です" << std::endl;
    return 1;
}

double avg = statcpp::mean(data.begin(), data.end());
```

### ソート忘れ

中央値や四分位数を計算する関数はソート済みのデータを要求します:

```cpp
std::vector<double> data = {5.0, 2.0, 8.0, 1.0, 3.0};

// ソートが必要
std::sort(data.begin(), data.end());

double med = statcpp::median(data.begin(), data.end());
```

## 次のステップ

- 各関数の詳細な仕様は [API リファレンス](API_REFERENCE.md) を参照してください
- 基本的な使い方は [使い方ガイド](USAGE.md) を参照してください
- テストの実行方法は [ビルドガイド](BUILDING.md) を参照してください
