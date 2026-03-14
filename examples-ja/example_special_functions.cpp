/**
 * @file example_special_functions.cpp
 * @brief 特殊関数のサンプルコード
 *
 * ガンマ関数、ベータ関数、誤差関数、正則化不完全ガンマ関数、
 * 正則化不完全ベータ関数等の特殊関数の使用例を示します。
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include "statcpp/special_functions.hpp"

int main() {
    std::cout << "=== 特殊関数の例 ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);

    // ============================================================================
    // 1. ガンマ関数とその対数
    // ============================================================================
    std::cout << "\n1. ガンマ関数とその対数" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> gamma_values = {1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, 2.5};

    std::cout << "ガンマ関数の値:" << std::endl;
    for (double x : gamma_values) {
        double gamma_x = statcpp::tgamma(x);
        double lgamma_x = statcpp::lgamma(x);
        std::cout << "  Γ(" << std::setw(4) << x << ") = " << std::setw(12) << gamma_x
                  << ",  log Γ(" << std::setw(4) << x << ") = " << std::setw(12) << lgamma_x
                  << std::endl;
    }

    std::cout << "\n注意: 正の整数に対して Γ(n) = (n-1)!" << std::endl;
    std::cout << "  Γ(1) = 0! = " << statcpp::tgamma(1.0) << std::endl;
    std::cout << "  Γ(2) = 1! = " << statcpp::tgamma(2.0) << std::endl;
    std::cout << "  Γ(3) = 2! = " << statcpp::tgamma(3.0) << std::endl;
    std::cout << "  Γ(4) = 3! = " << statcpp::tgamma(4.0) << std::endl;
    std::cout << "  Γ(5) = 4! = " << statcpp::tgamma(5.0) << std::endl;

    // ============================================================================
    // 2. ベータ関数
    // ============================================================================
    std::cout << "\n2. ベータ関数" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<std::pair<double, double>> beta_params = {
        {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0},
        {1.0, 2.0}, {2.0, 3.0}, {0.5, 0.5}
    };

    std::cout << "ベータ関数の値:" << std::endl;
    for (const auto& pair : beta_params) {
        double a = pair.first;
        double b = pair.second;
        double beta_ab = statcpp::beta(a, b);
        double lbeta_ab = statcpp::lbeta(a, b);
        std::cout << "  B(" << std::setw(4) << a << ", " << std::setw(4) << b << ") = "
                  << std::setw(12) << beta_ab
                  << ",  log B(" << std::setw(4) << a << ", " << std::setw(4) << b << ") = "
                  << std::setw(12) << lbeta_ab << std::endl;
    }

    std::cout << "\n注意: B(a,b) = Γ(a)Γ(b)/Γ(a+b)" << std::endl;

    // ============================================================================
    // 3. 正則化不完全ベータ関数
    // ============================================================================
    std::cout << "\n3. 正則化不完全ベータ関数" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::cout << "様々な値に対する I_x(a, b):" << std::endl;
    double a = 2.0, b = 3.0;
    std::vector<double> x_values = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

    std::cout << "\na = " << a << ", b = " << b << " の場合:" << std::endl;
    for (double x : x_values) {
        double betainc_val = statcpp::betainc(a, b, x);
        std::cout << "  I_{" << std::setw(3) << x << "}(" << a << ", " << b << ") = "
                  << std::setw(10) << betainc_val << std::endl;
    }

    // 逆関数のテスト
    std::cout << "\n逆関数のテスト (betaincinv):" << std::endl;
    std::vector<double> p_values = {0.1, 0.25, 0.5, 0.75, 0.9};
    for (double p : p_values) {
        double x_inv = statcpp::betaincinv(a, b, p);
        double p_check = statcpp::betainc(a, b, x_inv);
        std::cout << "  p = " << std::setw(4) << p << " -> x = " << std::setw(10) << x_inv
                  << " -> I_x = " << std::setw(10) << p_check << std::endl;
    }

    // ============================================================================
    // 4. 誤差関数
    // ============================================================================
    std::cout << "\n4. 誤差関数" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> erf_values = {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0};

    std::cout << "誤差関数の値:" << std::endl;
    for (double x : erf_values) {
        double erf_x = statcpp::erf(x);
        double erfc_x = statcpp::erfc(x);
        std::cout << "  erf(" << std::setw(5) << x << ") = " << std::setw(10) << erf_x
                  << ",  erfc(" << std::setw(5) << x << ") = " << std::setw(10) << erfc_x
                  << std::endl;
    }

    std::cout << "\n注意: erf(x) + erfc(x) = 1" << std::endl;
    std::cout << "  erf(1) + erfc(1) = " << (statcpp::erf(1.0) + statcpp::erfc(1.0)) << std::endl;

    // ============================================================================
    // 5. 標準正規分布の累積分布関数
    // ============================================================================
    std::cout << "\n5. 標準正規分布の累積分布関数" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::cout << "Φ(x) の値:" << std::endl;
    std::vector<double> norm_values = {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0};

    for (double x : norm_values) {
        double phi_x = statcpp::norm_cdf(x);
        std::cout << "  Φ(" << std::setw(5) << x << ") = " << std::setw(10) << phi_x << std::endl;
    }

    std::cout << "\n注意: Φ(0) = 0.5, Φ(-x) = 1 - Φ(x)" << std::endl;
    std::cout << "  Φ(-1.96) + Φ(1.96) = "
              << (statcpp::norm_cdf(-1.96) + statcpp::norm_cdf(1.96)) << std::endl;

    // ============================================================================
    // 6. 標準正規分布の逆累積分布関数（分位点）
    // ============================================================================
    std::cout << "\n6. 標準正規分布の逆累積分布関数（分位点）" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> quantile_probs = {0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99};

    std::cout << "Φ⁻¹(p) の値（分位点）:" << std::endl;
    for (double p : quantile_probs) {
        double q = statcpp::norm_quantile(p);
        double p_check = statcpp::norm_cdf(q);
        std::cout << "  Φ⁻¹(" << std::setw(5) << p << ") = " << std::setw(10) << q
                  << " -> Φ(" << std::setw(10) << q << ") = " << std::setw(10) << p_check
                  << std::endl;
    }

    std::cout << "\n一般的な分位点:" << std::endl;
    std::cout << "  90% 信頼区間: ±" << statcpp::norm_quantile(0.95) << std::endl;
    std::cout << "  95% 信頼区間: ±" << statcpp::norm_quantile(0.975) << std::endl;
    std::cout << "  99% 信頼区間: ±" << statcpp::norm_quantile(0.995) << std::endl;

    // ============================================================================
    // 7. 正則化不完全ガンマ関数
    // ============================================================================
    std::cout << "\n7. 正則化不完全ガンマ関数" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    double gamma_a = 2.5;
    std::vector<double> gamma_x = {0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0};

    std::cout << "a = " << gamma_a << " の場合の P(a, x) と Q(a, x):" << std::endl;
    for (double x : gamma_x) {
        double p_val = statcpp::gammainc_lower(gamma_a, x);
        double q_val = statcpp::gammainc_upper(gamma_a, x);
        std::cout << "  P(" << gamma_a << ", " << std::setw(4) << x << ") = " << std::setw(10) << p_val
                  << ",  Q(" << gamma_a << ", " << std::setw(4) << x << ") = " << std::setw(10) << q_val
                  << ",  P+Q = " << std::setw(10) << (p_val + q_val) << std::endl;
    }

    std::cout << "\n注意: P(a,x) + Q(a,x) = 1" << std::endl;

    // 逆関数のテスト
    std::cout << "\n逆関数のテスト (gammainc_lower_inv):" << std::endl;
    std::vector<double> gamma_p = {0.1, 0.25, 0.5, 0.75, 0.9};
    for (double p : gamma_p) {
        double x_inv = statcpp::gammainc_lower_inv(gamma_a, p);
        double p_check = statcpp::gammainc_lower(gamma_a, x_inv);
        std::cout << "  p = " << std::setw(4) << p << " -> x = " << std::setw(10) << x_inv
                  << " -> P(a,x) = " << std::setw(10) << p_check << std::endl;
    }

    // ============================================================================
    // 8. 定数の確認
    // ============================================================================
    std::cout << "\n8. 数学定数" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::cout << std::setprecision(15);
    std::cout << "π          = " << statcpp::pi << std::endl;
    std::cout << "√2         = " << statcpp::sqrt_2 << std::endl;
    std::cout << "√(2π)      = " << statcpp::sqrt_2_pi << std::endl;
    std::cout << "log(√(2π)) = " << statcpp::log_sqrt_2_pi << std::endl;

    // ============================================================================
    // 9. 実用例：カイ二乗分布のCDF
    // ============================================================================
    std::cout << std::setprecision(6);
    std::cout << "\n9. 実用例：カイ二乗分布のCDF" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    double df = 5.0;  // 自由度
    std::vector<double> chi2_values = {0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0};

    std::cout << "自由度 df = " << df << " のカイ二乗分布のCDF:" << std::endl;
    std::cout << "(P(df/2, x/2) 公式を使用)" << std::endl;
    for (double x : chi2_values) {
        // カイ二乗分布のCDF = P(df/2, x/2)
        double cdf = statcpp::gammainc_lower(df / 2.0, x / 2.0);
        std::cout << "  χ²(" << std::setw(5) << x << "; df=" << df << ") = " << std::setw(10) << cdf
                  << std::endl;
    }

    // ============================================================================
    // 10. 実用例：ベータ分布のCDF
    // ============================================================================
    std::cout << "\n10. 実用例：ベータ分布のCDF" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    double beta_a = 2.0, beta_b = 5.0;
    std::vector<double> beta_x = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

    std::cout << "α = " << beta_a << ", β = " << beta_b << " のベータ分布のCDF:" << std::endl;
    for (double x : beta_x) {
        double cdf = statcpp::betainc(beta_a, beta_b, x);
        std::cout << "  F(" << std::setw(3) << x << "; α=" << beta_a << ", β=" << beta_b << ") = "
                  << std::setw(10) << cdf << std::endl;
    }

    std::cout << "\n=== 例の実行が正常に完了しました ===" << std::endl;

    return 0;
}
