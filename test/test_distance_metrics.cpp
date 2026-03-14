#include <gtest/gtest.h>
#include "statcpp/distance_metrics.hpp"
#include <vector>
#include <cmath>

// ============================================================================
// Euclidean Distance Tests
// ============================================================================

/**
 * @brief 既知の値: (0,0) -> (3,4) = 5
 * @test ピタゴラスの定理による既知の距離を検証する
 */
TEST(EuclideanDistanceTest, KnownValue) {
    std::vector<double> a = {0.0, 0.0};
    std::vector<double> b = {3.0, 4.0};
    EXPECT_DOUBLE_EQ(statcpp::euclidean_distance(a.begin(), a.end(), b.begin(), b.end()), 5.0);
}

/**
 * @brief 同一の点間の距離は0
 * @test 同じベクトル間のユークリッド距離が0になることを検証する
 */
TEST(EuclideanDistanceTest, IdenticalPoints) {
    std::vector<double> a = {1.0, 2.0, 3.0};
    EXPECT_DOUBLE_EQ(statcpp::euclidean_distance(a.begin(), a.end(), a.begin(), a.end()), 0.0);
}

/**
 * @brief 異なる長さの系列で例外を投げる
 * @test 長さが異なる2つの系列に対して std::invalid_argument が投げられることを検証する
 */
TEST(EuclideanDistanceTest, DifferentLengths) {
    std::vector<double> a = {1.0, 2.0};
    std::vector<double> b = {1.0, 2.0, 3.0};
    EXPECT_THROW(statcpp::euclidean_distance(a.begin(), a.end(), b.begin(), b.end()),
                 std::invalid_argument);
}

/**
 * @brief 空の系列で例外を投げる
 * @test 空の系列に対して std::invalid_argument が投げられることを検証する
 */
TEST(EuclideanDistanceTest, EmptySequence) {
    std::vector<double> a;
    std::vector<double> b;
    EXPECT_THROW(statcpp::euclidean_distance(a.begin(), a.end(), b.begin(), b.end()),
                 std::invalid_argument);
}

/**
 * @brief Projection 付きのユークリッド距離
 * @test カスタム射影関数を使用した距離計算を検証する
 */
TEST(EuclideanDistanceTest, WithProjection) {
    struct Point { double x; };
    std::vector<Point> a = {{0.0}, {0.0}};
    std::vector<Point> b = {{3.0}, {4.0}};
    auto proj = [](const Point& p) { return p.x; };
    EXPECT_DOUBLE_EQ(statcpp::euclidean_distance(a.begin(), a.end(), b.begin(), b.end(), proj, proj), 5.0);
}

// ============================================================================
// Manhattan Distance Tests
// ============================================================================

/**
 * @brief 既知の値: (0,0) -> (3,4) = 7
 * @test L1ノルムによる既知の距離を検証する
 */
TEST(ManhattanDistanceTest, KnownValue) {
    std::vector<double> a = {0.0, 0.0};
    std::vector<double> b = {3.0, 4.0};
    EXPECT_DOUBLE_EQ(statcpp::manhattan_distance(a.begin(), a.end(), b.begin(), b.end()), 7.0);
}

/**
 * @brief 同一の点間の距離は0
 * @test 同じベクトル間のマンハッタン距離が0になることを検証する
 */
TEST(ManhattanDistanceTest, IdenticalPoints) {
    std::vector<double> a = {1.0, 2.0, 3.0};
    EXPECT_DOUBLE_EQ(statcpp::manhattan_distance(a.begin(), a.end(), a.begin(), a.end()), 0.0);
}

/**
 * @brief 異なる長さの系列で例外を投げる
 * @test 長さが異なる2つの系列に対して std::invalid_argument が投げられることを検証する
 */
TEST(ManhattanDistanceTest, DifferentLengths) {
    std::vector<double> a = {1.0, 2.0};
    std::vector<double> b = {1.0, 2.0, 3.0};
    EXPECT_THROW(statcpp::manhattan_distance(a.begin(), a.end(), b.begin(), b.end()),
                 std::invalid_argument);
}

/**
 * @brief 空の系列で例外を投げる
 * @test 空の系列に対して std::invalid_argument が投げられることを検証する
 */
TEST(ManhattanDistanceTest, EmptySequence) {
    std::vector<double> a;
    std::vector<double> b;
    EXPECT_THROW(statcpp::manhattan_distance(a.begin(), a.end(), b.begin(), b.end()),
                 std::invalid_argument);
}

/**
 * @brief 負の値を含むマンハッタン距離
 * @test 負の値を含むベクトルに対する距離計算を検証する
 */
TEST(ManhattanDistanceTest, NegativeValues) {
    std::vector<double> a = {-1.0, -2.0};
    std::vector<double> b = {3.0, 4.0};
    // |(-1)-3| + |(-2)-4| = 4 + 6 = 10
    EXPECT_DOUBLE_EQ(statcpp::manhattan_distance(a.begin(), a.end(), b.begin(), b.end()), 10.0);
}

// ============================================================================
// Cosine Similarity Tests
// ============================================================================

/**
 * @brief 平行なベクトルのコサイン類似度は1
 * @test 同方向のベクトルに対してコサイン類似度が1になることを検証する
 */
TEST(CosineSimilarityTest, ParallelVectors) {
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {2.0, 4.0, 6.0};
    EXPECT_NEAR(statcpp::cosine_similarity(a.begin(), a.end(), b.begin(), b.end()), 1.0, 1e-10);
}

/**
 * @brief 直交ベクトルのコサイン類似度は0
 * @test 直交するベクトルに対してコサイン類似度が0になることを検証する
 */
TEST(CosineSimilarityTest, OrthogonalVectors) {
    std::vector<double> a = {1.0, 0.0};
    std::vector<double> b = {0.0, 1.0};
    EXPECT_NEAR(statcpp::cosine_similarity(a.begin(), a.end(), b.begin(), b.end()), 0.0, 1e-10);
}

/**
 * @brief 逆平行なベクトルのコサイン類似度は-1
 * @test 逆方向のベクトルに対してコサイン類似度が-1になることを検証する
 */
TEST(CosineSimilarityTest, AntiParallelVectors) {
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {-1.0, -2.0, -3.0};
    EXPECT_NEAR(statcpp::cosine_similarity(a.begin(), a.end(), b.begin(), b.end()), -1.0, 1e-10);
}

/**
 * @brief ゼロベクトルで例外を投げる
 * @test ゼロベクトルに対して std::invalid_argument が投げられることを検証する
 */
TEST(CosineSimilarityTest, ZeroVector) {
    std::vector<double> a = {1.0, 2.0};
    std::vector<double> zero = {0.0, 0.0};
    EXPECT_THROW(statcpp::cosine_similarity(a.begin(), a.end(), zero.begin(), zero.end()),
                 std::invalid_argument);
    EXPECT_THROW(statcpp::cosine_similarity(zero.begin(), zero.end(), a.begin(), a.end()),
                 std::invalid_argument);
}

/**
 * @brief 空の系列で例外を投げる
 * @test 空の系列に対して std::invalid_argument が投げられることを検証する
 */
TEST(CosineSimilarityTest, EmptySequence) {
    std::vector<double> a;
    std::vector<double> b;
    EXPECT_THROW(statcpp::cosine_similarity(a.begin(), a.end(), b.begin(), b.end()),
                 std::invalid_argument);
}

/**
 * @brief 異なる長さの系列で例外を投げる
 * @test 長さが異なる2つの系列に対して std::invalid_argument が投げられることを検証する
 */
TEST(CosineSimilarityTest, DifferentLengths) {
    std::vector<double> a = {1.0, 2.0};
    std::vector<double> b = {1.0, 2.0, 3.0};
    EXPECT_THROW(statcpp::cosine_similarity(a.begin(), a.end(), b.begin(), b.end()),
                 std::invalid_argument);
}

// ============================================================================
// Cosine Distance Tests
// ============================================================================

/**
 * @brief コサイン距離 = 1 - コサイン類似度
 * @test コサイン距離がコサイン類似度の補数であることを検証する
 */
TEST(CosineDistanceTest, IsOneMinusSimilarity) {
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {4.0, 5.0, 6.0};
    double similarity = statcpp::cosine_similarity(a.begin(), a.end(), b.begin(), b.end());
    double distance = statcpp::cosine_distance(a.begin(), a.end(), b.begin(), b.end());
    EXPECT_NEAR(distance, 1.0 - similarity, 1e-10);
}

/**
 * @brief 同方向のベクトルのコサイン距離は0
 * @test 平行なベクトル間のコサイン距離が0になることを検証する
 */
TEST(CosineDistanceTest, ParallelVectorsZeroDistance) {
    std::vector<double> a = {1.0, 2.0};
    std::vector<double> b = {3.0, 6.0};
    EXPECT_NEAR(statcpp::cosine_distance(a.begin(), a.end(), b.begin(), b.end()), 0.0, 1e-10);
}

/**
 * @brief 逆方向のベクトルのコサイン距離は2
 * @test 逆平行なベクトル間のコサイン距離が2になることを検証する
 */
TEST(CosineDistanceTest, AntiParallelVectorsMaxDistance) {
    std::vector<double> a = {1.0, 0.0};
    std::vector<double> b = {-1.0, 0.0};
    EXPECT_NEAR(statcpp::cosine_distance(a.begin(), a.end(), b.begin(), b.end()), 2.0, 1e-10);
}

/**
 * @brief 空の系列で例外を投げる
 * @test 空の系列に対して std::invalid_argument が投げられることを検証する
 */
TEST(CosineDistanceTest, EmptySequence) {
    std::vector<double> a;
    std::vector<double> b;
    EXPECT_THROW(statcpp::cosine_distance(a.begin(), a.end(), b.begin(), b.end()),
                 std::invalid_argument);
}

// ============================================================================
// Mahalanobis Distance Tests
// ============================================================================

/**
 * @brief 単位共分散行列でユークリッド距離と一致する
 * @test 単位行列を共分散行列とした場合にユークリッド距離と等しくなることを検証する
 */
TEST(MahalanobisDistanceTest, IdentityCovarianceEqualsEuclidean) {
    std::vector<double> x = {3.0, 4.0};
    std::vector<double> mean = {0.0, 0.0};
    std::vector<std::vector<double>> identity = {{1.0, 0.0}, {0.0, 1.0}};
    double result = statcpp::mahalanobis_distance(x, mean, identity);
    double expected = std::sqrt(3.0 * 3.0 + 4.0 * 4.0);  // = 5.0
    EXPECT_NEAR(result, expected, 1e-10);
}

/**
 * @brief 非対角共分散行列での既知の値
 * @test 相関のある共分散行列に対する正しい距離計算を検証する
 */
TEST(MahalanobisDistanceTest, NonDiagonalCovariance) {
    std::vector<double> x = {1.0, 1.0};
    std::vector<double> mean = {0.0, 0.0};
    // 共分散行列 [2, 1; 1, 2], 逆行列 = (1/3) * [2, -1; -1, 2]
    // d^T * S^-1 * d = (1/3)(2*1 - 1*1 - 1*1 + 2*1) = (1/3)(2) = 2/3
    std::vector<std::vector<double>> cov = {{2.0, 1.0}, {1.0, 2.0}};
    double result = statcpp::mahalanobis_distance(x, mean, cov);
    double expected = std::sqrt(2.0 / 3.0);
    EXPECT_NEAR(result, expected, 1e-10);
}

/**
 * @brief 特異行列で例外を投げる
 * @test 行列式が0の共分散行列に対して std::invalid_argument が投げられることを検証する
 */
TEST(MahalanobisDistanceTest, SingularCovarianceThrows) {
    std::vector<double> x = {1.0, 1.0};
    std::vector<double> mean = {0.0, 0.0};
    std::vector<std::vector<double>> singular = {{1.0, 1.0}, {1.0, 1.0}};
    EXPECT_THROW(statcpp::mahalanobis_distance(x, mean, singular), std::invalid_argument);
}

/**
 * @brief 次元不一致で例外を投げる
 * @test x と mean の次元が異なる場合に std::invalid_argument が投げられることを検証する
 */
TEST(MahalanobisDistanceTest, DimensionMismatchThrows) {
    std::vector<double> x = {1.0, 2.0};
    std::vector<double> mean = {0.0};
    std::vector<std::vector<double>> cov = {{1.0, 0.0}, {0.0, 1.0}};
    EXPECT_THROW(statcpp::mahalanobis_distance(x, mean, cov), std::invalid_argument);
}

/**
 * @brief 非2次元で例外を投げる
 * @test 2次元以外の入力に対して std::invalid_argument が投げられることを検証する
 */
TEST(MahalanobisDistanceTest, NonTwoDimensionalThrows) {
    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> mean = {0.0, 0.0, 0.0};
    std::vector<std::vector<double>> cov = {{1.0, 0.0}, {0.0, 1.0}};
    EXPECT_THROW(statcpp::mahalanobis_distance(x, mean, cov), std::invalid_argument);
}

/**
 * @brief 同一の点の距離は0
 * @test x と mean が等しい場合にマハラノビス距離が0になることを検証する
 */
TEST(MahalanobisDistanceTest, SamePointZeroDistance) {
    std::vector<double> x = {3.0, 4.0};
    std::vector<double> mean = {3.0, 4.0};
    std::vector<std::vector<double>> cov = {{2.0, 0.5}, {0.5, 3.0}};
    EXPECT_NEAR(statcpp::mahalanobis_distance(x, mean, cov), 0.0, 1e-10);
}

// ============================================================================
// Minkowski Distance Tests
// ============================================================================

/**
 * @brief p=1 でマンハッタン距離と一致する
 * @test Minkowski距離(p=1)がマンハッタン距離と等しくなることを検証する
 */
TEST(MinkowskiDistanceTest, P1MatchesManhattan) {
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {4.0, 6.0, 8.0};
    double minkowski = statcpp::minkowski_distance(a.begin(), a.end(), b.begin(), b.end(), 1.0);
    double manhattan = statcpp::manhattan_distance(a.begin(), a.end(), b.begin(), b.end());
    EXPECT_NEAR(minkowski, manhattan, 1e-10);
}

/**
 * @brief p=2 でユークリッド距離と一致する
 * @test Minkowski距離(p=2)がユークリッド距離と等しくなることを検証する
 */
TEST(MinkowskiDistanceTest, P2MatchesEuclidean) {
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {4.0, 6.0, 8.0};
    double minkowski = statcpp::minkowski_distance(a.begin(), a.end(), b.begin(), b.end(), 2.0);
    double euclidean = statcpp::euclidean_distance(a.begin(), a.end(), b.begin(), b.end());
    EXPECT_NEAR(minkowski, euclidean, 1e-10);
}

/**
 * @brief p=3 での既知の値
 * @test L3ノルムによる既知の距離を検証する
 */
TEST(MinkowskiDistanceTest, P3KnownValue) {
    std::vector<double> a = {0.0, 0.0};
    std::vector<double> b = {3.0, 4.0};
    // (3^3 + 4^3)^(1/3) = (27 + 64)^(1/3) = 91^(1/3)
    double expected = std::pow(91.0, 1.0 / 3.0);
    EXPECT_NEAR(statcpp::minkowski_distance(a.begin(), a.end(), b.begin(), b.end(), 3.0),
                expected, 1e-10);
}

/**
 * @brief p<1 で例外を投げる
 * @test p が 1 未満の場合に std::invalid_argument が投げられることを検証する
 */
TEST(MinkowskiDistanceTest, PLessThanOneThrows) {
    std::vector<double> a = {1.0, 2.0};
    std::vector<double> b = {3.0, 4.0};
    EXPECT_THROW(statcpp::minkowski_distance(a.begin(), a.end(), b.begin(), b.end(), 0.5),
                 std::invalid_argument);
    EXPECT_THROW(statcpp::minkowski_distance(a.begin(), a.end(), b.begin(), b.end(), 0.0),
                 std::invalid_argument);
    EXPECT_THROW(statcpp::minkowski_distance(a.begin(), a.end(), b.begin(), b.end(), -1.0),
                 std::invalid_argument);
}

/**
 * @brief 空の系列で例外を投げる
 * @test 空の系列に対して std::invalid_argument が投げられることを検証する
 */
TEST(MinkowskiDistanceTest, EmptySequence) {
    std::vector<double> a;
    std::vector<double> b;
    EXPECT_THROW(statcpp::minkowski_distance(a.begin(), a.end(), b.begin(), b.end(), 2.0),
                 std::invalid_argument);
}

/**
 * @brief 異なる長さの系列で例外を投げる
 * @test 長さが異なる2つの系列に対して std::invalid_argument が投げられることを検証する
 */
TEST(MinkowskiDistanceTest, DifferentLengths) {
    std::vector<double> a = {1.0, 2.0};
    std::vector<double> b = {1.0, 2.0, 3.0};
    EXPECT_THROW(statcpp::minkowski_distance(a.begin(), a.end(), b.begin(), b.end(), 2.0),
                 std::invalid_argument);
}

// ============================================================================
// Chebyshev Distance Tests
// ============================================================================

/**
 * @brief 既知の値: 最大差分
 * @test L-infinity ノルムが各成分の最大絶対差であることを検証する
 */
TEST(ChebyshevDistanceTest, KnownValue) {
    std::vector<double> a = {1.0, 5.0, 3.0};
    std::vector<double> b = {4.0, 2.0, 8.0};
    // max(|1-4|, |5-2|, |3-8|) = max(3, 3, 5) = 5
    EXPECT_DOUBLE_EQ(statcpp::chebyshev_distance(a.begin(), a.end(), b.begin(), b.end()), 5.0);
}

/**
 * @brief 同一の点間の距離は0
 * @test 同じベクトル間のチェビシェフ距離が0になることを検証する
 */
TEST(ChebyshevDistanceTest, IdenticalPoints) {
    std::vector<double> a = {1.0, 2.0, 3.0};
    EXPECT_DOUBLE_EQ(statcpp::chebyshev_distance(a.begin(), a.end(), a.begin(), a.end()), 0.0);
}

/**
 * @brief 1次元のチェビシェフ距離
 * @test 1要素の系列に対する距離計算を検証する
 */
TEST(ChebyshevDistanceTest, SingleDimension) {
    std::vector<double> a = {3.0};
    std::vector<double> b = {7.0};
    EXPECT_DOUBLE_EQ(statcpp::chebyshev_distance(a.begin(), a.end(), b.begin(), b.end()), 4.0);
}

/**
 * @brief 空の系列で例外を投げる
 * @test 空の系列に対して std::invalid_argument が投げられることを検証する
 */
TEST(ChebyshevDistanceTest, EmptySequence) {
    std::vector<double> a;
    std::vector<double> b;
    EXPECT_THROW(statcpp::chebyshev_distance(a.begin(), a.end(), b.begin(), b.end()),
                 std::invalid_argument);
}

/**
 * @brief 異なる長さの系列で例外を投げる
 * @test 長さが異なる2つの系列に対して std::invalid_argument が投げられることを検証する
 */
TEST(ChebyshevDistanceTest, DifferentLengths) {
    std::vector<double> a = {1.0, 2.0};
    std::vector<double> b = {1.0, 2.0, 3.0};
    EXPECT_THROW(statcpp::chebyshev_distance(a.begin(), a.end(), b.begin(), b.end()),
                 std::invalid_argument);
}

/**
 * @brief Projection 付きのチェビシェフ距離
 * @test カスタム射影関数を使用した距離計算を検証する
 */
TEST(ChebyshevDistanceTest, WithProjection) {
    struct Point { double x; };
    std::vector<Point> a = {{1.0}, {5.0}, {3.0}};
    std::vector<Point> b = {{4.0}, {2.0}, {8.0}};
    auto proj = [](const Point& p) { return p.x; };
    EXPECT_DOUBLE_EQ(statcpp::chebyshev_distance(a.begin(), a.end(), b.begin(), b.end(), proj, proj), 5.0);
}

// ============================================================================
// Mathematical Identity Tests
// ============================================================================

/**
 * @brief 距離の非負性
 * @test 全ての距離関数が非負の値を返すことを検証する
 */
TEST(DistanceIdentityTest, NonNegativity) {
    std::vector<double> a = {1.0, -2.0, 3.0};
    std::vector<double> b = {-4.0, 5.0, -6.0};
    EXPECT_GE(statcpp::euclidean_distance(a.begin(), a.end(), b.begin(), b.end()), 0.0);
    EXPECT_GE(statcpp::manhattan_distance(a.begin(), a.end(), b.begin(), b.end()), 0.0);
    EXPECT_GE(statcpp::chebyshev_distance(a.begin(), a.end(), b.begin(), b.end()), 0.0);
    EXPECT_GE(statcpp::minkowski_distance(a.begin(), a.end(), b.begin(), b.end(), 3.0), 0.0);
}

/**
 * @brief 対称性: d(a,b) = d(b,a)
 * @test 距離関数が対称的であることを検証する
 */
TEST(DistanceIdentityTest, Symmetry) {
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {4.0, 6.0, 8.0};
    EXPECT_DOUBLE_EQ(statcpp::euclidean_distance(a.begin(), a.end(), b.begin(), b.end()),
                     statcpp::euclidean_distance(b.begin(), b.end(), a.begin(), a.end()));
    EXPECT_DOUBLE_EQ(statcpp::manhattan_distance(a.begin(), a.end(), b.begin(), b.end()),
                     statcpp::manhattan_distance(b.begin(), b.end(), a.begin(), a.end()));
    EXPECT_DOUBLE_EQ(statcpp::chebyshev_distance(a.begin(), a.end(), b.begin(), b.end()),
                     statcpp::chebyshev_distance(b.begin(), b.end(), a.begin(), a.end()));
}

/**
 * @brief チェビシェフ距離 <= ユークリッド距離 <= マンハッタン距離
 * @test Lp ノルムの順序関係を検証する
 */
TEST(DistanceIdentityTest, LpNormOrdering) {
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {4.0, 6.0, 8.0};
    double chebyshev = statcpp::chebyshev_distance(a.begin(), a.end(), b.begin(), b.end());
    double euclidean = statcpp::euclidean_distance(a.begin(), a.end(), b.begin(), b.end());
    double manhattan = statcpp::manhattan_distance(a.begin(), a.end(), b.begin(), b.end());
    EXPECT_LE(chebyshev, euclidean);
    EXPECT_LE(euclidean, manhattan);
}
