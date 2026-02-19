# インストール

statcpp は C++17 ヘッダーオンリーライブラリです。ヘッダーファイルをインクルードパスに追加するだけで使用できます。

英語版と日本語コメント版の両方が利用可能です。お好みのバージョンをお選びください。

## 必要な環境

- C++17 対応コンパイラ
  - GCC 7.0 以上
  - Clang 5.0 以上
  - Apple Clang 9.0 以上
- CMake 3.14 以上（オプション、テストやサンプルをビルドする場合）

## 動作確認環境

- macOS + Apple Clang 17.0.0
- Ubuntu 24.04 ARM64 + GCC

## インストール方法

### 方法1: ヘッダーオンリーとして直接使用

最もシンプルな方法です。`include/`（英語版）または `include-ja/`（日本語版）ディレクトリをプロジェクトにコピーして使用します。

```bash
# statcpp を任意の場所にクローン
git clone https://github.com/yourusername/statcpp.git

# include ディレクトリをプロジェクトにコピー
# 日本語版
cp -r statcpp/include-ja /path/to/your/project/include

# 英語版
cp -r statcpp/include /path/to/your/project/
```

コンパイル時にインクルードパスを指定:

```bash
g++ -std=c++17 -I/path/to/your/project/include your_program.cpp -o your_program
```

### 方法2: CMake でインストール

システム全体または特定の場所にインストールして使用する方法です。

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/statcpp.git
cd statcpp

# ビルドディレクトリを作成してインストール（日本語版）
mkdir build
cd build
cmake .. -DSTATCPP_USE_JAPANESE=ON
sudo cmake --install .

# 英語版をインストールする場合（デフォルト）
cmake ..
sudo cmake --install .
```

デフォルトでは `/usr/local/include/` にインストールされます。

カスタムインストール先を指定する場合:

```bash
cmake -DCMAKE_INSTALL_PREFIX=/your/custom/path ..
cmake --install .
```

#### CMake オプション

| オプション                   | デフォルト | 説明                                                    |
| ---------------------------- | ---------- | ------------------------------------------------------- |
| `STATCPP_USE_JAPANESE`       | OFF        | 日本語コメント版ヘッダーを使用                          |
| `STATCPP_BUILD_TESTS`        | ON         | テストスイートをビルド                                  |
| `STATCPP_ENABLE_SANITIZERS`  | OFF        | AddressSanitizer と UndefinedBehaviorSanitizer を有効化 |

### 方法3: CMake プロジェクトでサブディレクトリとして使用

既存の CMake プロジェクトにサブディレクトリとして追加する方法です。

```cmake
# CMakeLists.txt
add_subdirectory(external/statcpp)
target_link_libraries(your_target PRIVATE statcpp)
```

### 方法4: CMake の FetchContent を使用

CMake 3.14 以降では、FetchContent を使って自動的にダウンロードして使用できます。

```cmake
include(FetchContent)

FetchContent_Declare(
    statcpp
    GIT_REPOSITORY https://github.com/yourusername/statcpp.git
    GIT_TAG        main  # または特定のタグ/コミット
)
FetchContent_MakeAvailable(statcpp)

target_link_libraries(your_target PRIVATE statcpp)
```

## インストール後の使用方法

インストール後、ヘッダーファイルは `/usr/local/include/statcpp/`（またはカスタムプレフィックス）に配置されます。

`statcpp/` プレフィックス付きでヘッダーをインクルードします：

```bash
g++ -std=c++17 your_program.cpp -o your_program
```

```cpp
#include "statcpp/basic_statistics.hpp"
```

## インストールの確認

以下の簡単なプログラムでインストールを確認できます。

```cpp
#include <iostream>
#include <vector>
#include "statcpp/basic_statistics.hpp"

int main() {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    double avg = statcpp::mean(data.begin(), data.end());
    std::cout << "平均: " << avg << std::endl;  // 出力: 平均: 3
    return 0;
}
```

コンパイルと実行:

```bash
g++ -std=c++17 -I/path/to/statcpp/include test.cpp -o test
./test
```

## トラブルシューティング

### コンパイラが C++17 に対応していない

最新版のコンパイラにアップグレードしてください。

```bash
# macOS (Homebrew)
brew install gcc

# Ubuntu/Debian
sudo apt-get install g++-9

# 特定バージョンを指定してコンパイル
g++-9 -std=c++17 your_program.cpp -o your_program
```

### インクルードパスが見つからない

コンパイル時に `-I` オプションでパスを明示的に指定してください。

```bash
g++ -std=c++17 -I/usr/local/include your_program.cpp -o your_program
```

### CMake でライブラリが見つからない

`CMAKE_PREFIX_PATH` を設定してください。

```bash
cmake -DCMAKE_PREFIX_PATH=/your/custom/install/path ..
```
