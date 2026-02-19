# statcpp — Statistics Library for C++

C++17 ヘッダーオンリー統計ライブラリ

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)

[English](README.md)

## 概要

statcpp は C++17 で書かれたヘッダーオンリーの統計ライブラリです。31 個のヘッダーファイルに 524 個の公開関数を提供し、基本的な統計量から高度な統計的検定、回帰分析まで、幅広い統計機能をカバーします。758 件の単体テストと R 4.4.2 に対する 167 件の数値検証チェックを含みます。

### 主な特徴

- **524 個の公開関数**: 31 モジュールにわたる包括的な統計機能
- **ヘッダーオンリー**: ビルド不要、インクルードするだけで使用可能
- **C++17 標準準拠**: モダンな C++ の機能を活用
- **STL スタイル**: イテレータベースの直感的な API
- **射影サポート**: 構造体のメンバーなどを直接処理
- **包括的なテスト**: Google Test による 758 件の単体テスト、R 4.4.2 に対する 167 件の数値検証チェック
- **クロスプラットフォーム**: macOS、Linux で動作確認済み

### 提供機能

- **基本統計量**: 平均、中央値、最頻値、分散、標準偏差など
- **順序統計量**: 四分位数、パーセンタイル、五数要約
- **相関分析**: Pearson, Spearman, Kendall の相関係数
- **確率分布**: 正規分布、t分布、カイ二乗分布、F分布、二項分布、ポアソン分布など
- **仮説検定**: t検定、z検定、F検定、カイ二乗検定、Wilcoxon検定、Mann-Whitney検定など
- **効果量**: Cohen's d, Hedges' g, イータ二乗など
- **回帰分析**: 単回帰、重回帰、ロジスティック回帰
- **分散分析**: 一元配置、二元配置、反復測定 ANOVA
- **リサンプリング**: ブートストラップ、ジャックナイフ、置換検定
- **検出力分析**: サンプルサイズ計算と検出力分析
- **距離メトリクス**: ユークリッド距離、マンハッタン距離、コサイン類似度など
- **クラスタリング**: k-means、階層的クラスタリング

## クイックスタート

### インストール

```bash
# リポジトリをクローン
git clone https://github.com/mitsuruk/statcpp.git

# ヘッダーファイルをインクルードパスに追加
# 方法1: システムにインストール（日本語版）
cd statcpp
mkdir build && cd build
cmake .. -DSTATCPP_USE_JAPANESE=ON
sudo cmake --install .

# 方法1b: 英語版をインストール（デフォルト）
cmake ..
sudo cmake --install .

# 方法2: プロジェクトにコピー
cp -r statcpp/include-ja /your/project/     # 日本語版
cp -r statcpp/include /your/project/        # 英語版
```

詳細は [インストールガイド](docs-ja/INSTALLATION.md) を参照してください。

### 基本的な使い方

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"
#include "statcpp/order_statistics.hpp"

int main() {
    std::vector<double> data = {5.0, 2.0, 8.0, 1.0, 3.0, 7.0, 4.0};

    // 基本統計量
    double avg = statcpp::mean(data.begin(), data.end());
    double sd = statcpp::stddev(data.begin(), data.end());

    std::cout << "平均: " << avg << std::endl;        // 4.285...
    std::cout << "標準偏差: " << sd << std::endl;      // 2.429...

    // 順序統計量（ソートが必要）
    std::sort(data.begin(), data.end());
    double median = statcpp::median(data.begin(), data.end());
    auto q = statcpp::quartiles(data.begin(), data.end());

    std::cout << "中央値: " << median << std::endl;    // 4.0
    std::cout << "第1四分位数: " << q.Q1 << std::endl; // 2.5
    std::cout << "第3四分位数: " << q.Q3 << std::endl; // 6.5

    return 0;
}
```

コンパイルと実行:

```bash
g++ -std=c++17 -I/path/to/statcpp/include example.cpp -o example
./example
```

### 射影を使った例

構造体のメンバーを直接処理できます:

```cpp
#include "statcpp/basic_statistics.hpp"
#include <vector>

struct Product {
    std::string name;
    double price;
};

int main() {
    std::vector<Product> products = {
        {"Apple", 120.0},
        {"Banana", 80.0},
        {"Orange", 100.0}
    };

    // 価格の平均を計算
    double avg_price = statcpp::mean(
        products.begin(),
        products.end(),
        [](const Product& p) { return p.price; }
    );

    std::cout << "平均価格: " << avg_price << std::endl;  // 100.0
    return 0;
}
```

### 仮説検定の例

```cpp
#include "statcpp/parametric_tests.hpp"
#include <vector>

int main() {
    std::vector<double> group1 = {23, 21, 19, 24, 20};
    std::vector<double> group2 = {31, 28, 30, 29, 32};

    // 2標本t検定
    auto result = statcpp::t_test_two_sample(
        group1.begin(), group1.end(),
        group2.begin(), group2.end()
    );

    std::cout << "t統計量: " << result.statistic << std::endl;
    std::cout << "p値: " << result.p_value << std::endl;
    std::cout << "自由度: " << result.df << std::endl;

    if (result.p_value < 0.05) {
        std::cout << "有意差あり（p < 0.05）" << std::endl;
    }

    return 0;
}
```

## モジュール一覧

| モジュール | ヘッダーファイル | 内容 |
|-----------|----------------|------|
| 基本統計量 | `basic_statistics.hpp` | 平均、中央値、最頻値など |
| 散布度 | `dispersion_spread.hpp` | 分散、標準偏差、範囲など |
| 順序統計量 | `order_statistics.hpp` | 四分位数、パーセンタイルなど |
| 分布の形状 | `shape_of_distribution.hpp` | 歪度、尖度 |
| 相関・共分散 | `correlation_covariance.hpp` | 相関係数、共分散 |
| 度数分布 | `frequency_distribution.hpp` | ヒストグラム、度数表 |
| 特殊関数 | `special_functions.hpp` | ガンマ関数、誤差関数など |
| 乱数生成 | `random_engine.hpp` | 乱数生成エンジン |
| 連続分布 | `continuous_distributions.hpp` | 正規分布、t分布など |
| 離散分布 | `discrete_distributions.hpp` | 二項分布、ポアソン分布など |
| 推定 | `estimation.hpp` | 信頼区間の計算 |
| パラメトリック検定 | `parametric_tests.hpp` | t検定、z検定など |
| ノンパラメトリック検定 | `nonparametric_tests.hpp` | Wilcoxon検定など |
| 効果量 | `effect_size.hpp` | Cohen's d など |
| リサンプリング | `resampling.hpp` | ブートストラップなど |
| 検出力分析 | `power_analysis.hpp` | サンプルサイズ計算 |
| 線形回帰 | `linear_regression.hpp` | 単回帰、重回帰 |
| 分散分析 | `anova.hpp` | ANOVA |
| 一般化線形モデル | `glm.hpp` | ロジスティック回帰など |
| 距離メトリクス | `distance_metrics.hpp` | ユークリッド距離など |
| 数値ユーティリティ | `numerical_utils.hpp` | 数値計算ヘルパー |

その他: 多変量解析、時系列分析、クラスタリング、生存時間解析なども含まれています。

詳細は [API リファレンス](docs-ja/API_REFERENCE.md) を参照してください。

## ドキュメント

### ガイド

- **[インストール](docs-ja/INSTALLATION.md)** - インストール方法と環境設定
- **[使い方](docs-ja/USAGE.md)** - 基本的な使い方と共通仕様
- **[サンプルコード](docs-ja/EXAMPLES.md)** - 実用的なコード例
- **[API リファレンス](docs-ja/API_REFERENCE.md)** - 全モジュールと関数の概要
- **[ビルドとテスト](docs-ja/BUILDING.md)** - テストとサンプルのビルド方法
- **[貢献ガイド](docs-ja/CONTRIBUTING.md)** - プロジェクトへの貢献方法
- **[変更履歴](docs-ja/CHANGELOG.md)** - バージョン履歴
- **[TODO](todo.md)** - 開発予定・改善項目

### API ドキュメント

詳細な API ドキュメントは Doxygen で生成できます:

```bash
# Doxygen のインストール
brew install doxygen  # macOS
sudo apt-get install doxygen  # Ubuntu/Debian

# ドキュメント生成
./generate_docs.sh

# ブラウザで開く
open doc/html/index.html  # macOS
xdg-open doc/html/index.html  # Linux
```

## 動作確認環境

- macOS + Apple Clang 17.0.0
- macOS + GCC 15 (Homebrew)
- Ubuntu 24.04 ARM64 + GCC 13.3.0

## 開発の目的

以前から C++ の開発作業の中で統計的な計算をすることがあり、ある程度のコードのストックがありましたが、バラバラのコードを集めて簡易的な統計ライブラリとしてヘッダーのみで使用できる関数群を作成しました。

ターゲットプログラミング言語は C++17 で OS に依存しないコードを想定しています。
別途 Rust 版も作成予定です。

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 貢献

プロジェクトへの貢献を歓迎します。バグ報告、機能要望、プルリクエストなど、お気軽にどうぞ。

詳細は [貢献ガイド](docs-ja/CONTRIBUTING.md) を参照してください。

## サポート

- **Issue**: バグ報告や機能要望は [GitHub Issues](https://github.com/yourusername/statcpp/issues) で
- **Discussion**: 質問や議論は [GitHub Discussions](https://github.com/yourusername/statcpp/discussions) で

## 謝辞

このプロジェクトの開発には以下のツールと AI を活用しています:

- **OpenAI ChatGPT 5.2** - ドキュメント類の構文確認、説明不足の確認
- **Claude Code for VS Code Opus 4.5** - Google Test 用コードの生成、サンプルコードの修正、リファクタリング
- **LM Studio google/gemma-2-27b** - ドキュメント類の構文確認、説明不足の確認
- **llmama.cpp/gemma-2-27b** - 統合ビルドとエラーログの管理

---

**注意**: このライブラリは数値安定性や極端なエッジケースへの対応において、商用の統計ソフトウェアと同等のレベルではありません。研究や本番環境で使用する場合は、結果を他のツールで検証することをお勧めします。
