# 基本的な使い方

statcpp ライブラリの基本的な使用方法とパターンを説明します。

## クイックスタート

```cpp
#include <iostream>
#include <vector>
#include "statcpp/basic_statistics.hpp"

int main() {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};

    // 平均を計算
    double avg = statcpp::mean(data.begin(), data.end());
    std::cout << "平均: " << avg << std::endl;  // 3.0

    return 0;
}
```

### インクルードパスの設定

statcpp はヘッダーオンリーライブラリです。ヘッダーは `include/statcpp/` ディレクトリ（日本語コメント版は `include-ja/statcpp/`）に配置されています。コンパイラの `-I` フラグで `include/` ディレクトリをインクルードパスに追加してください:

```bash
# インクルードパスを指定してコンパイル
g++ -std=c++17 -I/path/to/statcpp/include your_program.cpp -o your_program
```

`statcpp/` プレフィックス付きでヘッダーをインクルードします:

```cpp
#include "statcpp/basic_statistics.hpp"
```

CMake で `find_package(statcpp)` または `add_subdirectory` を使用する場合、インクルードパスは自動的に設定されます。

## 共通仕様

### イテレータインターフェース

すべての関数は、半開区間 $[first,\, last)$ を表すイテレータペア `(first, last)` により範囲を受け取ります。
`std::vector`、`std::array`、組み込み配列（`T a[N]`）、生ポインタ範囲（`T* first, T* last`）など、RandomAccessIterator を提供する任意のシーケンスで使用可能です。

**イテレータカテゴリ**: すべての関数は **RandomAccessIterator** を要求します。
これは `*(first + i)` によるランダムアクセスや `std::distance` の $O(1)$ 計算を前提としているためです。
`std::forward_list` や入力ストリームイテレータなど、RandomAccessIterator でないイテレータは使用できません。

```cpp
#include "statcpp/basic_statistics.hpp"
#include <vector>
#include <array>

// std::vector を使用
std::vector<double> vec = {1.0, 2.0, 3.0};
double m1 = statcpp::mean(vec.begin(), vec.end());

// std::array を使用
std::array<int, 5> arr = {1, 2, 3, 4, 5};
double m2 = statcpp::mean(arr.begin(), arr.end());

// C スタイル配列を使用
double data[] = {1.0, 2.0, 3.0};
double m3 = statcpp::mean(data, data + 3);
```

### 射影 (Projection)

多くの関数は射影関数（ラムダ式等の callable）を追加引数として受け取るオーバーロードを持ちます。
射影関数 $f$ が渡された場合、各要素 $x_i$ に対して $f(x_i)$ を評価し、その結果に対して統計量を計算します。

対応する関数:
- `sum`, `mean`, `median`, `mode`, `geometric_mean`, `harmonic_mean`, `trimmed_mean`
- `range`, `population_variance`, `sample_variance`, `variance`
- `population_stddev`, `sample_stddev`, `stddev`
- `coefficient_of_variation`, `iqr`, `mean_absolute_deviation`
- `minimum`, `maximum`, `quartiles`, `percentile`, `five_number_summary`

```cpp
#include "statcpp/basic_statistics.hpp"
#include <vector>

struct Product {
    std::string name;
    double price;
};

std::vector<Product> products = {
    {"Apple", 120.0},
    {"Banana", 80.0},
    {"Orange", 100.0}
};

// 価格の平均を計算（射影を使用）
double avg_price = statcpp::mean(
    products.begin(),
    products.end(),
    [](const Product& p) { return p.price; }
);
// avg_price = 100.0
```

### 事前計算済み平均 (Pre-computed Mean)

分散や標準偏差を計算する関数は、事前計算済みの平均値を `double precomputed_mean` として受け取るオーバーロードを持ちます。
複数の統計量を効率的に計算する場合に有用です。

対応する関数:
- `population_variance`, `sample_variance`, `variance`
- `population_stddev`, `sample_stddev`, `stddev`
- `coefficient_of_variation`, `mean_absolute_deviation`

射影版では `(first, last, proj, precomputed_mean)` の引数順となります。

```cpp
#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"
#include <vector>

std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};

// 平均を一度だけ計算
double avg = statcpp::mean(data.begin(), data.end());

// 事前計算済みの平均を使用して分散と標準偏差を計算
double var = statcpp::variance(data.begin(), data.end(), avg);
double sd = statcpp::stddev(data.begin(), data.end(), avg);
```

### ソート済み範囲

`median`, `trimmed_mean`, `iqr`, `quartiles`, `percentile`, `five_number_summary` はソート済みの範囲を前提とします。
呼び出し側であらかじめ昇順にソートしたシーケンス（またはそのイテレータ範囲）を渡してください。

ソート順序は `std::sort` と同様に **strict weak ordering**（厳密弱順序）を満たす比較に基づきます。

```cpp
#include "statcpp/basic_statistics.hpp"
#include "statcpp/order_statistics.hpp"
#include <vector>
#include <algorithm>

std::vector<double> data = {5.0, 2.0, 8.0, 1.0, 3.0};

// ソートが必要
std::sort(data.begin(), data.end());

// ソート済みのデータで中央値を計算
double med = statcpp::median(data.begin(), data.end());
```

射影版では、**射影関数 $f$ の戻り値**が昇順となるように要素が並んでいる必要があります。

```cpp
struct Product {
    std::string name;
    double price;
};

std::vector<Product> products = {
    {"Apple", 120.0},
    {"Banana", 80.0},
    {"Orange", 100.0}
};

// 射影キー price でソートしてから渡す
std::sort(products.begin(), products.end(),
          [](const Product& a, const Product& b) { return a.price < b.price; });

auto q = statcpp::quartiles(products.begin(), products.end(),
                            [](const Product& p) { return p.price; });
```

### 分位点の線形補間法

`iqr`, `quartiles`, `percentile`, `five_number_summary` で用いる分位点の計算は、
線形補間法（R `type=7` / Excel `QUARTILE.INC` / `PERCENTILE.INC` 相当）に基づきます。

パラメータ $p\ （0 \leq p \leq 1）$に対し、0始まりインデックスで:

$$
\text{index} = p \times (n - 1)
$$

$$
lo = \lfloor \text{index} \rfloor, \quad frac = \text{index} - lo
$$

$$
Q = x[lo] \times (1 - frac) + x[lo + 1] \times frac
$$

**端点の取り扱い**: $lo + 1 \geq n$ の場合（$p = 1$ すなわち $lo = n - 1$ を含む）、$Q = x[lo]$ を返します。
これにより $p = 0$ で最小値、$p = 1$ で最大値が返されます。

### 例外処理

不正な引数（範囲外のパラメータ、計算不能な条件等）や、統計量の定義上空の範囲が許容されない場合は `std::invalid_argument` を送出します。
メッセージには名前空間修飾付き関数名を含みます（例: `"statcpp::mean: empty range"`）。

ただし `sum` と `count` は空の範囲を許容し、それぞれ `value_type{}` と `0` を返します。

```cpp
#include "statcpp/basic_statistics.hpp"
#include <vector>
#include <stdexcept>

std::vector<double> empty_data;

try {
    double avg = statcpp::mean(empty_data.begin(), empty_data.end());
} catch (const std::invalid_argument& e) {
    std::cerr << "エラー: " << e.what() << std::endl;
    // 出力: エラー: statcpp::mean: empty range
}

// sum と count は例外を投げない
double s = statcpp::sum(empty_data.begin(), empty_data.end());  // 0.0
std::size_t n = statcpp::count(empty_data.begin(), empty_data.end());  // 0
```

### 数値安定性

分散・標準偏差の計算は二段階法（two-pass algorithm: 先に平均を求め、次に偏差の二乗和を求める）で実装しています。
Welford のオンラインアルゴリズム等は使用していないため、極端にスケールが異なるデータでは桁落ちが生じる可能性があります。
高精度が必要な場合は呼び出し側でデータの正規化を検討してください。

```cpp
#include "statcpp/dispersion_spread.hpp"
#include <vector>

// スケールが大きく異なるデータ
std::vector<double> data = {1e10, 1e10 + 1, 1e10 + 2};

// 正規化してから計算
double mean_val = statcpp::mean(data.begin(), data.end());
std::vector<double> normalized;
for (double x : data) {
    normalized.push_back(x - mean_val);
}

double var = statcpp::variance(normalized.begin(), normalized.end());
```

## 複数モジュールの組み合わせ

複数のヘッダーを組み合わせて使用する例:

```cpp
#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"
#include "statcpp/order_statistics.hpp"
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<double> data = {5.0, 2.0, 8.0, 1.0, 3.0, 7.0, 4.0};

    // 基本統計量
    double avg = statcpp::mean(data.begin(), data.end());
    std::cout << "平均: " << avg << std::endl;

    // 散布度
    double sd = statcpp::stddev(data.begin(), data.end());
    std::cout << "標準偏差: " << sd << std::endl;

    // 順序統計量（ソートが必要）
    std::sort(data.begin(), data.end());
    double med = statcpp::median(data.begin(), data.end());
    std::cout << "中央値: " << med << std::endl;

    auto q = statcpp::quartiles(data.begin(), data.end());
    std::cout << "第1四分位数: " << q.Q1 << std::endl;
    std::cout << "第2四分位数: " << q.Q2 << std::endl;
    std::cout << "第3四分位数: " << q.Q3 << std::endl;

    return 0;
}
```

## 次のステップ

- より詳しい関数リファレンスは [API リファレンス](API_REFERENCE.md) を参照してください
- 実用的なコード例は [サンプルコード](EXAMPLES.md) を参照してください
- ビルドやテストについては [ビルドガイド](BUILDING.md) を参照してください
