# ビルドとテスト

statcpp はヘッダーオンリーライブラリのため、ライブラリ自体のビルドは不要です。
このドキュメントでは、テストのビルドと実行、サンプルプログラムのビルド、ドキュメント生成について説明します。

## 必要な環境

### 基本要件

- C++17 対応コンパイラ
  - GCC 7.0 以上
  - Clang 5.0 以上
  - Apple Clang 9.0 以上

### テストのビルドに必要

- CMake 3.14 以上
- Google Test（事前にシステムへのインストールが必要）

### ドキュメント生成に必要

- [Doxygen](https://www.doxygen.nl/)

## テストのビルドと実行

### 方法1: CMake を使用（推奨）

```bash
# プロジェクトルートディレクトリで実行
mkdir build
cd build

# CMake 設定（テストを有効化）
cmake -DSTATCPP_BUILD_TESTS=ON ..

# ビルド
cmake --build .

# テスト実行
ctest --output-on-failure
```

または:

```bash
# テストディレクトリから直接実行
cd test
mkdir build
cd build
cmake ..
cmake --build .
ctest --output-on-failure
```

### 方法2: 特定のテストのみ実行

```bash
cd build

# 特定のテストファイルを実行
./test/test_basic_statistics
./test/test_dispersion_spread
./test/test_order_statistics
# ...
```

### 利用可能なテスト

プロジェクトには以下のテストスイートが含まれています:

- `test_basic_statistics` - 基本統計量
- `test_dispersion_spread` - 散布度
- `test_order_statistics` - 順序統計量
- `test_shape_of_distribution` - 分布の形状
- `test_correlation_covariance` - 相関・共分散
- `test_frequency_distribution` - 度数分布
- `test_special_functions` - 特殊関数
- `test_random_engine` - 乱数生成
- `test_continuous_distributions` - 連続分布
- `test_discrete_distributions` - 離散分布
- `test_estimation` - 推定
- `test_parametric_tests` - パラメトリック検定
- `test_nonparametric_tests` - ノンパラメトリック検定
- `test_effect_size` - 効果量
- `test_resampling` - リサンプリング
- `test_power_analysis` - 検出力分析
- `test_linear_regression` - 線形回帰
- `test_anova` - 分散分析
- `test_glm` - 一般化線形モデル
- `test_model_selection` - モデル選択
- `test_multivariate` - 多変量解析
- `test_time_series` - 時系列分析
- `test_categorical` - カテゴリカルデータ分析
- `test_survival` - 生存時間解析
- `test_robust` - 頑健統計
- `test_clustering` - クラスタリング
- `test_data_wrangling` - データ変換
- `test_missing_data` - 欠損データ

## サンプルプログラムのビルド

### 方法1: シェルスクリプトを使用（macOS/Linux）

最も簡単な方法です:

```bash
cd examples-ja
./build.sh
```

このスクリプトは:
1. すべての `.cpp` ファイルをコンパイル
2. 実行ファイルを生成
3. 各プログラムを実行して出力を表示

### 方法2: CMake を使用

```bash
cd examples-ja
mkdir build
cd build
cmake ..
cmake --build .

# 実行
./example_basic_statistics
./example_dispersion_spread
# ...
```

### 方法3: 個別にコンパイル

```bash
cd examples-ja

# GCC
g++ -std=c++17 -I../include example_basic_statistics.cpp -o example_basic_statistics

# Clang
clang++ -std=c++17 -I../include example_basic_statistics.cpp -o example_basic_statistics

# Apple Clang (macOS)
clang++ -std=c++17 -I../include example_basic_statistics.cpp -o example_basic_statistics

# 実行
./example_basic_statistics
```

## ドキュメント生成

### Doxygen のインストール

```bash
# macOS (Homebrew)
brew install doxygen

# Ubuntu/Debian
sudo apt-get install doxygen

# Windows
# https://www.doxygen.nl/download.html からダウンロード
```

### ドキュメントの生成

```bash
# プロジェクトルートディレクトリで実行

# シェルスクリプトを使用（推奨）
./generate_docs.sh

# または直接 Doxygen を実行
doxygen Doxyfile
```

生成されたドキュメントは `doc/html/index.html` に出力されます。

### ドキュメントの閲覧

```bash
# macOS
open doc/html/index.html

# Linux
xdg-open doc/html/index.html

# Windows
start doc/html/index.html
```

## CMake ビルドオプション

### テストの無効化

```bash
cmake -DSTATCPP_BUILD_TESTS=OFF ..
```

### サニタイザの有効化

メモリエラーや未定義動作を検出するためのサニタイザを有効にできます:

```bash
cmake -DSTATCPP_ENABLE_SANITIZERS=ON ..
cmake --build .
ctest --output-on-failure
```

**注意**: サニタイザは開発・デバッグ時のみ使用し、本番環境では無効にしてください。

### カスタムコンパイラの指定

```bash
# GCC を指定
cmake -DCMAKE_CXX_COMPILER=g++-11 ..

# Clang を指定
cmake -DCMAKE_CXX_COMPILER=clang++-14 ..
```

### ビルドタイプの指定

```bash
# Debug ビルド（デバッグシンボル付き）
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release ビルド（最適化有効）
cmake -DCMAKE_BUILD_TYPE=Release ..

# RelWithDebInfo（最適化 + デバッグシンボル）
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
```

## トラブルシューティング

### CMake が見つからない

```bash
# macOS
brew install cmake

# Ubuntu/Debian
sudo apt-get install cmake

# または公式サイトからダウンロード
# https://cmake.org/download/
```

### Google Test が見つからない

Google Test はシステムに事前にインストールされている必要があります。

```bash
# Ubuntu/Debian
sudo apt-get install libgtest-dev

# macOS
brew install googletest

# Windows (vcpkg)
vcpkg install gtest
```

### コンパイラエラー: C++17 サポートがない

古いコンパイラを使用している可能性があります。最新版にアップグレードしてください:

```bash
# Ubuntu/Debian
sudo apt-get install g++-9

# macOS
brew install gcc
```

### テストが失敗する

1. **数値精度の問題**: 浮動小数点演算の丸め誤差により、環境によってテストが失敗することがあります。
2. **乱数のシード**: 乱数を使用するテストは、シードが異なると結果が変わる可能性があります。

詳細なエラー出力を確認:

```bash
ctest --output-on-failure --verbose
```

特定のテストを直接実行:

```bash
./test/test_basic_statistics --gtest_filter=MeanTest.*
```

### ドキュメント生成でワーニングが出る

Doxygen のワーニングは通常無視できます。エラーが発生した場合は:

1. Doxygen のバージョンを確認（1.9.0 以上推奨）
2. `Doxyfile` の設定を確認

## 継続的インテグレーション（CI）

プロジェクトに CI を設定する場合の参考:

### GitHub Actions の例

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: sudo apt-get install -y cmake g++
      - name: Build and test
        run: |
          mkdir build
          cd build
          cmake -DSTATCPP_BUILD_TESTS=ON ..
          cmake --build .
          ctest --output-on-failure
```

## 動作確認環境

本ライブラリの動作確認環境:

- **macOS + Apple Clang 17.0.0**
- **Ubuntu 24.04 ARM64 + GCC**

## 次のステップ

- サンプルコードの詳細は [サンプルコード](EXAMPLES.md) を参照してください
- コントリビューションガイドラインは [貢献ガイド](CONTRIBUTING.md) を参照してください
- API の詳細は [API リファレンス](API_REFERENCE.md) を参照してください
