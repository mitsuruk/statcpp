/**
 * @file example_continuous_distributions.cpp
 * @brief 連続確率分布のサンプルコード
 *
 * 正規分布、対数正規分布、t分布、カイ二乗分布、F分布、
 * 指数分布、ワイブル分布等の連続確率分布の使用例を示します。
 */

#include <iostream>
#include <iomanip>
#include "statcpp/continuous_distributions.hpp"

int main()
{
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "=== 連続分布のサンプル ===\n\n";

    // Normal Distribution
    std::cout << "1. 正規分布 (μ=0, σ=1)\n";
    std::cout << "   x=0 での確率密度: " << statcpp::normal_pdf(0.0) << "\n";
    std::cout << "   x=0 での累積分布: " << statcpp::normal_cdf(0.0) << "\n";
    std::cout << "   95パーセンタイル: " << statcpp::normal_quantile(0.95) << "\n";
    std::cout << "   ランダムサンプル: " << statcpp::normal_rand() << "\n\n";

    // Log-normal Distribution
    std::cout << "2. 対数正規分布 (μ=0, σ=1)\n";
    std::cout << "   x=1 での確率密度: " << statcpp::lognormal_pdf(1.0) << "\n";
    std::cout << "   x=1 での累積分布: " << statcpp::lognormal_cdf(1.0) << "\n";
    std::cout << "   中央値: " << statcpp::lognormal_quantile(0.5) << "\n";
    std::cout << "   ランダムサンプル: " << statcpp::lognormal_rand() << "\n\n";

    // t-Distribution
    std::cout << "3. スチューデントのt分布 (自由度=10)\n";
    std::cout << "   t=0 での確率密度: " << statcpp::t_pdf(0.0, 10.0) << "\n";
    std::cout << "   t=2 での累積分布: " << statcpp::t_cdf(2.0, 10.0) << "\n";
    std::cout << "   97.5パーセンタイル: " << statcpp::t_quantile(0.975, 10.0) << "\n";
    std::cout << "   ランダムサンプル: " << statcpp::t_rand(10.0) << "\n\n";

    // Chi-square Distribution
    std::cout << "4. カイ二乗分布 (自由度=5)\n";
    std::cout << "   x=5 での確率密度: " << statcpp::chisq_pdf(5.0, 5.0) << "\n";
    std::cout << "   x=5 での累積分布: " << statcpp::chisq_cdf(5.0, 5.0) << "\n";
    std::cout << "   95パーセンタイル: " << statcpp::chisq_quantile(0.95, 5.0) << "\n";
    std::cout << "   ランダムサンプル: " << statcpp::chisq_rand(5.0) << "\n\n";

    // F-Distribution
    std::cout << "5. F分布 (自由度1=5, 自由度2=10)\n";
    std::cout << "   x=1 での確率密度: " << statcpp::f_pdf(1.0, 5.0, 10.0) << "\n";
    std::cout << "   x=2 での累積分布: " << statcpp::f_cdf(2.0, 5.0, 10.0) << "\n";
    std::cout << "   95パーセンタイル: " << statcpp::f_quantile(0.95, 5.0, 10.0) << "\n";
    std::cout << "   ランダムサンプル: " << statcpp::f_rand(5.0, 10.0) << "\n\n";

    // Exponential Distribution
    std::cout << "6. 指数分布 (λ=2.0)\n";
    std::cout << "   x=0.5 での確率密度: " << statcpp::exponential_pdf(0.5, 2.0) << "\n";
    std::cout << "   x=0.5 での累積分布: " << statcpp::exponential_cdf(0.5, 2.0) << "\n";
    std::cout << "   中央値: " << statcpp::exponential_quantile(0.5, 2.0) << "\n";
    std::cout << "   ランダムサンプル: " << statcpp::exponential_rand(2.0) << "\n\n";

    // Weibull Distribution
    std::cout << "7. ワイブル分布 (形状=2.0, 尺度=1.0)\n";
    std::cout << "   x=1 での確率密度: " << statcpp::weibull_pdf(1.0, 2.0, 1.0) << "\n";
    std::cout << "   x=1 での累積分布: " << statcpp::weibull_cdf(1.0, 2.0, 1.0) << "\n";
    std::cout << "   中央値: " << statcpp::weibull_quantile(0.5, 2.0, 1.0) << "\n";
    std::cout << "   ランダムサンプル: " << statcpp::weibull_rand(2.0, 1.0) << "\n\n";

    return 0;
}
