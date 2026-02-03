# コントリビューションガイド

statcpp プロジェクトへの貢献を歓迎します。このドキュメントでは、プロジェクトへの貢献方法とガイドラインを説明します。

## 貢献の方法

### 1. バグ報告

バグを見つけた場合は、以下の情報を含めて Issue を作成してください:

- **環境情報**
  - OS とバージョン
  - コンパイラとバージョン
  - C++ 標準（C++17, C++20 など）
- **再現手順**
  - 最小限の再現可能なコード例
  - 期待される動作
  - 実際の動作
- **エラーメッセージ**
  - コンパイルエラーまたは実行時エラーの全文

### 2. 機能追加の提案

新しい機能や改善の提案は大歓迎です:

- **背景と動機**
  - なぜこの機能が必要か
  - どのようなユースケースで役立つか
- **提案内容**
  - API の設計案
  - 使用例のコードスニペット
- **代替案**
  - 他の実装方法の検討

### 3. プルリクエスト

コードの貢献は以下の手順で行ってください:

1. **フォークとクローン**
   ```bash
   # リポジトリをフォーク
   git clone https://github.com/yourusername/statcpp.git
   cd statcpp
   ```

2. **ブランチを作成**
   ```bash
   git checkout -b feature/your-feature-name
   # または
   git checkout -b fix/bug-description
   ```

3. **変更を加える**
   - コードを実装
   - テストを追加
   - ドキュメントを更新

4. **テストを実行**
   ```bash
   mkdir build
   cd build
   cmake -DSTATCPP_BUILD_TESTS=ON ..
   cmake --build .
   ctest --output-on-failure
   ```

5. **コミット**
   ```bash
   git add .
   git commit -m "Add: 新機能の簡潔な説明"
   ```

6. **プッシュとプルリクエスト**
   ```bash
   git push origin feature/your-feature-name
   ```
   GitHub でプルリクエストを作成

## コーディング規約

### ファイル構成

- **ヘッダーファイル**: `include/statcpp/` に配置
- **テストファイル**: `test/` に配置（`test_*.cpp` の形式）
- **サンプルファイル**: `examples-ja/` に配置（`example_*.cpp` の形式）

### コーディングスタイル

#### 命名規則

```cpp
// 名前空間: 小文字
namespace statcpp {

// 関数名: スネークケース（小文字 + アンダースコア）
double mean(Iterator first, Iterator last);
double standard_deviation(Iterator first, Iterator last);

// 構造体/クラス名: パスカルケース（各単語の先頭が大文字）
struct TestResult {
    double statistic;
    double p_value;
};

// 変数名: スネークケース
double sample_mean = 0.0;
std::size_t sample_size = 100;

// 定数: 大文字 + アンダースコア
constexpr double DEFAULT_ALPHA = 0.05;

// テンプレートパラメータ: パスカルケース
template <typename Iterator, typename Projection>
double mean(Iterator first, Iterator last, Projection proj);
}
```

#### インデントとフォーマット

```cpp
// インデント: スペース 4 つ
void example_function() {
    if (condition) {
        // コード
    }
}

// 波括弧の位置
void function() {  // 同じ行
    // コード
}

struct MyStruct {  // 同じ行
    int value;
};
```

#### ヘッダーファイルの構造

```cpp
#ifndef STATCPP_MODULE_NAME_HPP
#define STATCPP_MODULE_NAME_HPP

#include <vector>
#include <algorithm>
// その他の標準ライブラリ

namespace statcpp {

/**
 * @brief 関数の簡潔な説明
 *
 * 詳細な説明（必要に応じて数式や使用例を含める）
 *
 * @tparam Iterator ランダムアクセスイテレータ
 * @param first 範囲の先頭イテレータ
 * @param last 範囲の終端イテレータ
 * @return 計算結果
 * @throws std::invalid_argument 不正な入力の場合
 */
template <typename Iterator>
double function_name(Iterator first, Iterator last) {
    // 実装
}

}  // namespace statcpp

#endif  // STATCPP_MODULE_NAME_HPP
```

### ドキュメンテーション

すべての公開関数には Doxygen スタイルのコメントを付けてください:

```cpp
/**
 * @brief 算術平均を計算
 *
 * イテレータ範囲 [first, last) の要素の算術平均を計算します。
 *
 * 数式:
 * \f[
 * \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
 * \f]
 *
 * @tparam Iterator ランダムアクセスイテレータ
 * @param first 範囲の先頭
 * @param last 範囲の終端（要素を含まない）
 * @return 平均値（double 型）
 * @throws std::invalid_argument 範囲が空の場合
 *
 * 使用例:
 * @code
 * std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
 * double avg = statcpp::mean(data.begin(), data.end());
 * // avg = 3.0
 * @endcode
 */
template <typename Iterator>
double mean(Iterator first, Iterator last);
```

### テストの作成

すべての新機能にはテストを追加してください:

```cpp
#include <gtest/gtest.h>
#include "statcpp/module_name.hpp"
#include <vector>

// テストケース名: モジュール名 + Test
// テスト名: 機能の説明
TEST(ModuleNameTest, BasicFunctionality) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    double result = statcpp::function_name(data.begin(), data.end());
    EXPECT_DOUBLE_EQ(result, 3.0);
}

TEST(ModuleNameTest, EmptyRange) {
    std::vector<double> empty;
    EXPECT_THROW(
        statcpp::function_name(empty.begin(), empty.end()),
        std::invalid_argument
    );
}

TEST(ModuleNameTest, SingleElement) {
    std::vector<double> single = {42.0};
    double result = statcpp::function_name(single.begin(), single.end());
    EXPECT_DOUBLE_EQ(result, 42.0);
}

// 浮動小数点数の近似比較
TEST(ModuleNameTest, FloatingPointComparison) {
    std::vector<double> data = {1.1, 2.2, 3.3};
    double result = statcpp::function_name(data.begin(), data.end());
    EXPECT_NEAR(result, 2.2, 1e-10);  // 許容誤差 1e-10
}
```

### エラーハンドリング

```cpp
// 不正な入力には std::invalid_argument を投げる
template <typename Iterator>
double mean(Iterator first, Iterator last) {
    if (first == last) {
        throw std::invalid_argument("statcpp::mean: empty range");
    }
    // 実装
}

// エラーメッセージには名前空間付き関数名を含める
throw std::invalid_argument("statcpp::function_name: error description");
```

## 開発のベストプラクティス

### 1. 数値安定性

浮動小数点演算では数値安定性に注意してください:

```cpp
// 悪い例: 桁落ちが発生しやすい
double variance = (sum_of_squares / n) - (mean * mean);

// 良い例: 二段階法
double mean_val = mean(first, last);
double sum_sq_dev = 0.0;
for (auto it = first; it != last; ++it) {
    double dev = *it - mean_val;
    sum_sq_dev += dev * dev;
}
double variance = sum_sq_dev / n;
```

### 2. イテレータの要件

すべての関数は RandomAccessIterator を要求します:

```cpp
template <typename Iterator>
double function_name(Iterator first, Iterator last) {
    static_assert(
        std::is_base_of_v<
            std::random_access_iterator_tag,
            typename std::iterator_traits<Iterator>::iterator_category
        >,
        "RandomAccessIterator required"
    );
    // 実装
}
```

### 3. 射影のサポート

可能な限り射影バージョンを提供してください:

```cpp
// 基本版
template <typename Iterator>
double mean(Iterator first, Iterator last);

// 射影版
template <typename Iterator, typename Projection>
double mean(Iterator first, Iterator last, Projection proj) {
    // proj を使用して値を変換
}
```

### 4. パフォーマンス

- 不要なコピーを避ける（const 参照を使用）
- 可能な限り `inline` を使用
- テンプレートは完全にヘッダーに実装

```cpp
template <typename Iterator>
inline double mean(Iterator first, Iterator last) {
    // 実装
}
```

## サンプルコードの追加

新しい機能には使用例を追加してください:

1. `examples-ja/` に `example_your_feature.cpp` を作成
2. 実用的な使用例を含める
3. コメントで説明を追加

```cpp
#include <iostream>
#include <vector>
#include "statcpp/your_module.hpp"

int main() {
    std::cout << "=== Your Feature Example ===" << std::endl;

    // サンプルデータ
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};

    // 機能の使用例
    double result = statcpp::your_function(data.begin(), data.end());
    std::cout << "結果: " << result << std::endl;

    return 0;
}
```

## ドキュメントの更新

コードの変更に応じて、以下のドキュメントを更新してください:

- `docs/API_REFERENCE.md` - 新しい関数を追加
- `docs/EXAMPLES.md` - 新しいサンプルプログラムを追加
- `docs/CHANGELOG.md` - 変更履歴を記録

## コミットメッセージ

明確で簡潔なコミットメッセージを心がけてください:

```
Add: 新機能の追加
Fix: バグ修正
Update: 既存機能の改善
Refactor: リファクタリング
Test: テストの追加・修正
Docs: ドキュメントの更新
```

例:
```
Add: weighted_variance function to dispersion_spread module
Fix: numeric overflow in variance calculation for large datasets
Update: improve numerical stability of stddev function
Test: add edge case tests for empty ranges
Docs: update API reference with new functions
```

## レビュープロセス

プルリクエストは以下の点を確認します:

1. **コードの品質**
   - コーディング規約に準拠しているか
   - 適切なエラーハンドリングがあるか
2. **テスト**
   - すべてのテストが通るか
   - 新機能に対するテストがあるか
3. **ドキュメント**
   - Doxygen コメントがあるか
   - ドキュメントが更新されているか
4. **互換性**
   - 既存の API に破壊的変更がないか

## 質問やサポート

- **Issue**: バグや機能要望は GitHub Issues で
- **Discussion**: 一般的な質問や議論は GitHub Discussions で

## ライセンス

コントリビューションは、プロジェクトのライセンス（MIT License など）に従います。
プルリクエストを作成することで、あなたのコントリビューションがこのライセンスの下で配布されることに同意したものとみなされます。

## 謝辞

プロジェクトへの貢献に感謝します。あなたの貢献が statcpp をより良いライブラリにします。
