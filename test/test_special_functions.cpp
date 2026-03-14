#include <gtest/gtest.h>
#include "statcpp/special_functions.hpp"
#include <cmath>
#include <limits>

// ============================================================================
// Gamma Function Tests
// ============================================================================

/**
 * @brief Tests log-gamma function for positive integers
 * @test Verifies that lgamma produces correct values for positive integer arguments
 */
TEST(LgammaTest, PositiveIntegers) {
    // lgamma(n) = log((n-1)!)
    EXPECT_NEAR(statcpp::lgamma(1.0), 0.0, 1e-10);           // log(0!) = 0
    EXPECT_NEAR(statcpp::lgamma(2.0), 0.0, 1e-10);           // log(1!) = 0
    EXPECT_NEAR(statcpp::lgamma(3.0), std::log(2.0), 1e-10); // log(2!) = log(2)
    EXPECT_NEAR(statcpp::lgamma(4.0), std::log(6.0), 1e-10); // log(3!) = log(6)
    EXPECT_NEAR(statcpp::lgamma(5.0), std::log(24.0), 1e-10); // log(4!) = log(24)
}

/**
 * @brief Tests log-gamma function for half-integer values
 * @test Verifies that lgamma produces correct values for half-integer arguments
 */
TEST(LgammaTest, HalfIntegers) {
    // Γ(1/2) = √π
    EXPECT_NEAR(statcpp::lgamma(0.5), std::log(std::sqrt(statcpp::pi)), 1e-10);
    // Γ(3/2) = √π/2
    EXPECT_NEAR(statcpp::lgamma(1.5), std::log(std::sqrt(statcpp::pi) / 2.0), 1e-10);
}

/**
 * @brief Tests gamma function for positive integers
 * @test Verifies that tgamma produces correct factorial values for positive integer arguments
 */
TEST(TgammaTest, PositiveIntegers) {
    EXPECT_NEAR(statcpp::tgamma(1.0), 1.0, 1e-10);
    EXPECT_NEAR(statcpp::tgamma(2.0), 1.0, 1e-10);
    EXPECT_NEAR(statcpp::tgamma(3.0), 2.0, 1e-10);
    EXPECT_NEAR(statcpp::tgamma(4.0), 6.0, 1e-10);
    EXPECT_NEAR(statcpp::tgamma(5.0), 24.0, 1e-10);
}

/**
 * @brief Tests gamma function for half-integer values
 * @test Verifies that tgamma produces correct values for half-integer arguments
 */
TEST(TgammaTest, HalfIntegers) {
    EXPECT_NEAR(statcpp::tgamma(0.5), std::sqrt(statcpp::pi), 1e-10);
}

/**
 * @brief Tests gamma function error handling for invalid arguments
 * @test Verifies that gamma functions throw std::domain_error for non-positive integer arguments
 */
TEST(GammaTest, ThrowsOnNonPositiveInteger) {
    EXPECT_THROW(statcpp::lgamma(0.0), std::domain_error);
    EXPECT_THROW(statcpp::lgamma(-1.0), std::domain_error);
    EXPECT_THROW(statcpp::tgamma(0.0), std::domain_error);
    EXPECT_THROW(statcpp::tgamma(-2.0), std::domain_error);
}

// ============================================================================
// Beta Function Tests
// ============================================================================

/**
 * @brief Tests beta function for basic values
 * @test Verifies that beta function produces correct values for known argument pairs
 */
TEST(BetaTest, BasicValues) {
    // B(a, b) = Γ(a)Γ(b) / Γ(a+b)
    EXPECT_NEAR(statcpp::beta(1.0, 1.0), 1.0, 1e-10);
    EXPECT_NEAR(statcpp::beta(2.0, 2.0), 1.0/6.0, 1e-10);
    EXPECT_NEAR(statcpp::beta(1.0, 2.0), 0.5, 1e-10);
    EXPECT_NEAR(statcpp::beta(3.0, 2.0), 1.0/12.0, 1e-10);
}

/**
 * @brief Tests beta function symmetry property
 * @test Verifies that beta function is symmetric, i.e., B(a,b) = B(b,a)
 */
TEST(BetaTest, Symmetry) {
    EXPECT_NEAR(statcpp::beta(2.0, 3.0), statcpp::beta(3.0, 2.0), 1e-10);
    EXPECT_NEAR(statcpp::beta(0.5, 1.5), statcpp::beta(1.5, 0.5), 1e-10);
}

/**
 * @brief Tests beta function error handling for invalid arguments
 * @test Verifies that beta function throws std::domain_error for non-positive arguments
 */
TEST(BetaTest, ThrowsOnInvalidInput) {
    EXPECT_THROW(statcpp::beta(0.0, 1.0), std::domain_error);
    EXPECT_THROW(statcpp::beta(1.0, 0.0), std::domain_error);
    EXPECT_THROW(statcpp::beta(-1.0, 1.0), std::domain_error);
}

// ============================================================================
// Incomplete Beta Function Tests
// ============================================================================

/**
 * @brief Tests incomplete beta function at boundary values
 * @test Verifies that incomplete beta function returns 0 at x=0 and 1 at x=1
 */
TEST(BetaincTest, BoundaryValues) {
    EXPECT_DOUBLE_EQ(statcpp::betainc(2.0, 3.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(statcpp::betainc(2.0, 3.0, 1.0), 1.0);
}

/**
 * @brief Tests incomplete beta function symmetry relation
 * @test Verifies that I_x(a,b) = 1 - I_{1-x}(b,a)
 */
TEST(BetaincTest, SymmetryRelation) {
    // I_x(a, b) = 1 - I_{1-x}(b, a)
    double a = 2.0, b = 3.0, x = 0.3;
    EXPECT_NEAR(statcpp::betainc(a, b, x), 1.0 - statcpp::betainc(b, a, 1.0 - x), 1e-10);
}

/**
 * @brief Tests incomplete beta function for known values
 * @test Verifies that I_x(1,1) = x for uniform distribution
 */
TEST(BetaincTest, KnownValues) {
    // For a=b=1, I_x(1,1) = x (uniform distribution)
    EXPECT_NEAR(statcpp::betainc(1.0, 1.0, 0.5), 0.5, 1e-10);
    EXPECT_NEAR(statcpp::betainc(1.0, 1.0, 0.3), 0.3, 1e-10);
}

/**
 * @brief Tests inverse incomplete beta function
 * @test Verifies that betaincinv is the inverse of betainc
 */
TEST(BetaincinvTest, InverseProperty) {
    double a = 2.0, b = 3.0;
    double p = 0.4;
    double x = statcpp::betaincinv(a, b, p);
    EXPECT_NEAR(statcpp::betainc(a, b, x), p, 1e-8);
}

// ============================================================================
// Error Function Tests
// ============================================================================

/**
 * @brief Tests error function for known values
 * @test Verifies that error function produces correct values for known arguments
 */
TEST(ErfTest, KnownValues) {
    EXPECT_NEAR(statcpp::erf(0.0), 0.0, 1e-8);
    EXPECT_NEAR(statcpp::erf(1.0), 0.8427007929, 1e-5);
    EXPECT_NEAR(statcpp::erf(2.0), 0.9953222650, 1e-5);
}

/**
 * @brief Tests error function symmetry property
 * @test Verifies that erf(-x) = -erf(x)
 */
TEST(ErfTest, Symmetry) {
    EXPECT_NEAR(statcpp::erf(-1.0), -statcpp::erf(1.0), 1e-10);
    EXPECT_NEAR(statcpp::erf(-0.5), -statcpp::erf(0.5), 1e-10);
}

/**
 * @brief Tests complementary error function relation to erf
 * @test Verifies that erfc(x) = 1 - erf(x)
 */
TEST(ErfcTest, ComplementRelation) {
    EXPECT_NEAR(statcpp::erfc(1.0), 1.0 - statcpp::erf(1.0), 1e-10);
    EXPECT_NEAR(statcpp::erfc(0.5), 1.0 - statcpp::erf(0.5), 1e-10);
}

// ============================================================================
// Normal CDF and Quantile Tests
// ============================================================================

/**
 * @brief Tests normal cumulative distribution function for standard values
 * @test Verifies that normal CDF produces correct probabilities for known z-scores
 */
TEST(NormCdfTest, StandardValues) {
    EXPECT_NEAR(statcpp::norm_cdf(0.0), 0.5, 1e-8);
    EXPECT_NEAR(statcpp::norm_cdf(1.0), 0.8413447, 1e-5);
    EXPECT_NEAR(statcpp::norm_cdf(-1.0), 0.1586553, 1e-5);
    EXPECT_NEAR(statcpp::norm_cdf(1.96), 0.975, 1e-3);
}

/**
 * @brief Tests normal quantile function for standard values
 * @test Verifies that normal quantile produces correct z-scores for known probabilities
 */
TEST(NormQuantileTest, StandardValues) {
    EXPECT_NEAR(statcpp::norm_quantile(0.5), 0.0, 1e-10);
    EXPECT_NEAR(statcpp::norm_quantile(0.975), 1.96, 1e-2);
    EXPECT_NEAR(statcpp::norm_quantile(0.025), -1.96, 1e-2);
}

/**
 * @brief Tests that normal quantile is inverse of normal CDF
 * @test Verifies that norm_quantile is the inverse function of norm_cdf
 */
TEST(NormQuantileTest, InverseProperty) {
    double p = 0.7;
    double x = statcpp::norm_quantile(p);
    EXPECT_NEAR(statcpp::norm_cdf(x), p, 1e-6);
}

/**
 * @brief Tests normal quantile function at extreme probability values
 * @test Verifies that normal quantile returns infinity at p=0 and p=1
 */
TEST(NormQuantileTest, ExtremeValues) {
    EXPECT_EQ(statcpp::norm_quantile(0.0), -std::numeric_limits<double>::infinity());
    EXPECT_EQ(statcpp::norm_quantile(1.0), std::numeric_limits<double>::infinity());
}

// ============================================================================
// Incomplete Gamma Function Tests
// ============================================================================

/**
 * @brief Tests lower incomplete gamma function at boundary values
 * @test Verifies that lower incomplete gamma function returns 0 at x=0
 */
TEST(GammaincLowerTest, BoundaryValues) {
    EXPECT_DOUBLE_EQ(statcpp::gammainc_lower(2.0, 0.0), 0.0);
}

/**
 * @brief Tests lower incomplete gamma function for known values
 * @test Verifies that P(1,x) = 1 - e^(-x) for integer shape parameter
 */
TEST(GammaincLowerTest, KnownValues) {
    // For integer a, P(a, x) has known closed forms
    // P(1, x) = 1 - e^(-x)
    EXPECT_NEAR(statcpp::gammainc_lower(1.0, 1.0), 1.0 - std::exp(-1.0), 1e-10);
    EXPECT_NEAR(statcpp::gammainc_lower(1.0, 2.0), 1.0 - std::exp(-2.0), 1e-10);
}

/**
 * @brief Tests complement relation between lower and upper incomplete gamma
 * @test Verifies that P(a,x) + Q(a,x) = 1
 */
TEST(GammaincLowerTest, ComplementRelation) {
    double a = 2.0, x = 1.5;
    EXPECT_NEAR(statcpp::gammainc_lower(a, x) + statcpp::gammainc_upper(a, x), 1.0, 1e-10);
}

/**
 * @brief Tests inverse lower incomplete gamma function
 * @test Verifies that gammainc_lower_inv is the inverse of gammainc_lower
 */
TEST(GammaincLowerInvTest, InverseProperty) {
    double a = 3.0, p = 0.6;
    double x = statcpp::gammainc_lower_inv(a, p);
    EXPECT_NEAR(statcpp::gammainc_lower(a, x), p, 1e-8);
}

// ============================================================================
// erf 高精度テスト (NIST 参照値)
// ============================================================================

/**
 * @brief erf(x) の高精度テスト: x = 0.0
 * @test NIST 参照値との一致を絶対誤差 1e-14 で検証する
 */
TEST(ErfPrecisionTest, ZeroExact) {
    EXPECT_NEAR(statcpp::erf(0.0), 0.0, 1e-14);
}

/**
 * @brief erf(x) の高精度テスト: 小さい x (0 付近の値)
 * @test NIST 参照値との一致を絶対誤差 1e-14 で検証する
 */
TEST(ErfPrecisionTest, SmallArguments) {
    // erf(0.1) - NIST 参照値
    EXPECT_NEAR(statcpp::erf(0.1), 0.1124629160182848984, 1e-14);
    // erf(0.5) - NIST 参照値
    EXPECT_NEAR(statcpp::erf(0.5), 0.5204998778130465377, 1e-14);
}

/**
 * @brief erf(x) の高精度テスト: 1 に近い値を返す引数
 * @test NIST 参照値との相対誤差が 1e-14 未満であることを検証する
 */
TEST(ErfPrecisionTest, LargeArguments) {
    auto check_relative = [](double result, double ref, const char* label) {
        double rel_err = std::abs(result - ref) / std::abs(ref);
        EXPECT_LT(rel_err, 1e-14) << label << ": relative error = " << rel_err;
    };

    // erf(1.0) - NIST 参照値
    check_relative(statcpp::erf(1.0), 0.8427007929497148693, "erf(1.0)");
    // erf(2.0) - NIST 参照値
    check_relative(statcpp::erf(2.0), 0.9953222650189527342, "erf(2.0)");
    // erf(3.0) - NIST 参照値
    check_relative(statcpp::erf(3.0), 0.9999779095030014146, "erf(3.0)");
    // erf(5.0) - NIST 参照値
    check_relative(statcpp::erf(5.0), 0.9999999999984625402, "erf(5.0)");
}

// ============================================================================
// erfc 高精度テスト (NIST 参照値)
// ============================================================================

/**
 * @brief erfc(x) の高精度テスト: 中程度の x
 * @test NIST 参照値との相対誤差が 1e-13 未満であることを検証する
 */
TEST(ErfcPrecisionTest, ModerateArguments) {
    auto check_relative = [](double result, double ref, const char* label) {
        double rel_err = std::abs(result - ref) / std::abs(ref);
        EXPECT_LT(rel_err, 1e-13) << label << ": relative error = " << rel_err;
    };

    // erfc(0.1) - NIST 参照値
    check_relative(statcpp::erfc(0.1), 0.8875370839817151016, "erfc(0.1)");
    // erfc(1.0) - NIST 参照値
    check_relative(statcpp::erfc(1.0), 0.1572992070502851307, "erfc(1.0)");
}

/**
 * @brief erfc(x) の高精度テスト: 大きい x (裾の極小値)
 * @test NIST 参照値との相対誤差が 1e-13 未満であることを検証する.
 *       裾では値が非常に小さくなるため,相対誤差で評価する.
 */
TEST(ErfcPrecisionTest, TailArguments) {
    auto check_relative = [](double result, double ref, const char* label) {
        double rel_err = std::abs(result - ref) / std::abs(ref);
        // tail域では参照値の有効桁数が限られるため,1e-12 に緩和
        EXPECT_LT(rel_err, 1e-12) << label << ": relative error = " << rel_err;
    };

    // erfc(3.0) - NIST 参照値
    check_relative(statcpp::erfc(3.0), 2.209049699858544e-05, "erfc(3.0)");
    // erfc(4.0) - NIST 参照値
    check_relative(statcpp::erfc(4.0), 1.541725790028002e-08, "erfc(4.0)");
    // erfc(5.0) - NIST 参照値
    check_relative(statcpp::erfc(5.0), 1.537459794428e-12, "erfc(5.0)");
    // erfc(6.0) - NIST 参照値
    check_relative(statcpp::erfc(6.0), 2.151973671249e-17, "erfc(6.0)");
    // erfc(8.0) - NIST 参照値
    check_relative(statcpp::erfc(8.0), 1.1224297172983e-29, "erfc(8.0)");
}
