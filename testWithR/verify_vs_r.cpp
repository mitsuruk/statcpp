/**
 * @brief Verification program: compare statcpp output against R reference values.
 *
 * All R reference values were generated with R 4.4.2 using documented parameters.
 * This program checks each value against the R reference with a default tolerance
 * of 1e-6 (relative/absolute).
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <limits>

#include "statcpp/basic_statistics.hpp"
#include "statcpp/order_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"
#include "statcpp/shape_of_distribution.hpp"
#include "statcpp/correlation_covariance.hpp"
#include "statcpp/continuous_distributions.hpp"
#include "statcpp/discrete_distributions.hpp"
#include "statcpp/parametric_tests.hpp"
#include "statcpp/nonparametric_tests.hpp"
#include "statcpp/anova.hpp"
#include "statcpp/linear_regression.hpp"
#include "statcpp/glm.hpp"
#include "statcpp/effect_size.hpp"
#include "statcpp/estimation.hpp"
#include "statcpp/categorical.hpp"
#include "statcpp/survival.hpp"
#include "statcpp/power_analysis.hpp"
#include "statcpp/model_selection.hpp"
#include "statcpp/robust.hpp"
#include "statcpp/time_series.hpp"
#include "statcpp/multivariate.hpp"
#include "statcpp/data_wrangling.hpp"

static int g_pass = 0, g_fail = 0, g_section_fail = 0;
static const char* g_section = "";

static void section(const char* name) {
    if (g_section[0] && g_section_fail > 0) {
        printf("  >> %d FAILURES in %s\n", g_section_fail, g_section);
    }
    printf("\n=== %s ===\n", name);
    g_section = name;
    g_section_fail = 0;
}

static void check(const char* name, double cpp_val, double r_val, double tol = 1e-6) {
    if (std::isinf(cpp_val) && std::isinf(r_val) && ((cpp_val > 0) == (r_val > 0))) {
        g_pass++; return;
    }
    if (std::isnan(cpp_val) && std::isnan(r_val)) {
        g_pass++; return;
    }
    double diff = std::abs(cpp_val - r_val);
    if (!std::isnan(diff) && diff < tol) {
        g_pass++;
    } else {
        printf("  FAIL  %-55s cpp=%.15g  R=%.15g  diff=%.3e\n", name, cpp_val, r_val, diff);
        g_fail++;
        g_section_fail++;
    }
}

int main() {
    using namespace statcpp;

    // =========================================================================
    section("PARAMETRIC TESTS");
    // =========================================================================

    // One-sample t-test: R: t.test(x, mu=5.0)
    {
        std::vector<double> x = {5.1, 4.9, 5.3, 5.1, 4.8, 5.2, 5.0, 4.9};
        auto r = t_test(x.begin(), x.end(), 5.0);
        check("t_test(one-sample): t", r.statistic, 0.6294651818);
        check("t_test(one-sample): df", r.df, 7.0);
        check("t_test(one-sample): p", r.p_value, 0.5490293321);
    }

    // Two-sample t-test (equal var): R: t.test(g1, g2, var.equal=TRUE)
    {
        std::vector<double> g1 = {5.1, 4.9, 5.3, 5.1, 4.8};
        std::vector<double> g2 = {4.5, 4.7, 4.3, 4.6, 4.4};
        auto r = t_test_two_sample(g1.begin(), g1.end(), g2.begin(), g2.end());
        check("t_test_two_sample: t", r.statistic, 4.8107023544);
        check("t_test_two_sample: df", r.df, 8.0);
        check("t_test_two_sample: p", r.p_value, 0.0013371324);
    }

    // Welch t-test: R: t.test(g1, g2)
    {
        std::vector<double> g1 = {5.1, 4.9, 5.3, 5.1, 4.8};
        std::vector<double> g2 = {4.5, 4.7, 4.3, 4.6, 4.4};
        auto r = t_test_welch(g1.begin(), g1.end(), g2.begin(), g2.end());
        check("t_test_welch: t", r.statistic, 4.8107023544);
        check("t_test_welch: df", r.df, 7.6732721121, 1e-4);
        check("t_test_welch: p", r.p_value, 0.0015031235);
    }

    // Paired t-test: R: t.test(pre, post, paired=TRUE)
    {
        std::vector<double> pre = {5.1, 4.9, 5.3, 5.1, 4.8, 5.2};
        std::vector<double> post = {5.5, 5.2, 5.6, 5.3, 5.1, 5.4};
        auto r = t_test_paired(pre.begin(), pre.end(), post.begin(), post.end());
        check("t_test_paired: t", r.statistic, -9.2195444573);
        check("t_test_paired: df", r.df, 5.0);
        check("t_test_paired: p", r.p_value, 0.0002520663);
    }

    // F-test: R: var.test(g1, g2)
    {
        std::vector<double> g1 = {5.1, 4.9, 5.3, 5.1, 4.8};
        std::vector<double> g2 = {4.5, 4.7, 4.3, 4.6, 4.4};
        auto r = f_test(g1.begin(), g1.end(), g2.begin(), g2.end());
        check("f_test: F", r.statistic, 1.52);
        check("f_test: p", r.p_value, 0.6948693646);
    }

    // Chi-squared GOF: R: chisq.test(obs, p=rep(1/6,6))
    {
        std::vector<double> obs = {16, 18, 16, 14, 12, 12};
        std::vector<double> exp_v = {14.666666667, 14.666666667, 14.666666667,
                                     14.666666667, 14.666666667, 14.666666667};
        auto r = chisq_test_gof(obs.begin(), obs.end(), exp_v.begin(), exp_v.end());
        check("chisq_gof: X2", r.statistic, 2.0, 0.01);
        check("chisq_gof: df", r.df, 5.0);
        check("chisq_gof: p", r.p_value, 0.8491450361, 1e-4);
    }

    // Chi-squared independence: R: chisq.test(matrix(c(10,30,20,40),2,2), correct=FALSE)
    {
        std::vector<std::vector<double>> tbl = {{10, 20}, {30, 40}};
        auto r = chisq_test_independence(tbl);
        check("chisq_independence: X2", r.statistic, 0.7936507937, 1e-4);
        check("chisq_independence: df", r.df, 1.0);
        check("chisq_independence: p", r.p_value, 0.3729984836, 1e-4);
    }

    // Z-test for proportion: R: prop.test(45, 200, p=0.20, correct=FALSE)
    {
        auto r = z_test_proportion(45, 200, 0.20);
        // R: X2 = 0.78125, p = 0.376759 (prop.test returns X2 = z^2)
        check("z_test_prop: z^2 = X2", r.statistic * r.statistic, 0.78125, 0.01);
        check("z_test_prop: p", r.p_value, 0.376759117811582, 1e-4);
    }

    // Two-sample proportion test: R: prop.test(c(45,55), c(200,250), correct=FALSE)
    {
        auto r = z_test_proportion_two_sample(45, 200, 55, 250);
        check("z_test_prop_two: z^2 = X2", r.statistic * r.statistic, 0.0160714285714, 0.01);
        check("z_test_prop_two: p", r.p_value, 0.899119956774, 1e-3);
    }

    // Multiple testing corrections: R: p.adjust()
    {
        std::vector<double> pv = {0.01, 0.04, 0.03, 0.005, 0.15, 0.08};

        auto bonf = bonferroni_correction(pv);
        check("bonferroni[0]", bonf[0], 0.06);
        check("bonferroni[1]", bonf[1], 0.24);
        check("bonferroni[2]", bonf[2], 0.18);
        check("bonferroni[3]", bonf[3], 0.03);
        check("bonferroni[4]", bonf[4], 0.90);
        check("bonferroni[5]", bonf[5], 0.48);

        auto bh = benjamini_hochberg_correction(pv);
        check("BH[0]", bh[0], 0.030);
        check("BH[1]", bh[1], 0.060);
        check("BH[2]", bh[2], 0.060);
        check("BH[3]", bh[3], 0.030);
        check("BH[4]", bh[4], 0.150);
        check("BH[5]", bh[5], 0.096);

        // holm_correction: not implemented in statcpp
    }

    // =========================================================================
    section("NONPARAMETRIC TESTS");
    // =========================================================================

    // Wilcoxon signed-rank: R: wilcox.test(x, mu=5.0) default (correct=TRUE)
    {
        std::vector<double> x = {5.1, 4.9, 5.3, 5.1, 4.8, 5.2, 5.0, 4.9};
        auto r = wilcoxon_signed_rank_test(x.begin(), x.end(), 5.0);
        check("wilcoxon_signed_rank: V", r.statistic, 17.5);
        check("wilcoxon_signed_rank: p", r.p_value, 0.6049071549);
    }

    // Mann-Whitney U: R: wilcox.test(g1, g2)
    {
        std::vector<double> g1 = {5.1, 4.9, 5.3, 5.1, 4.8};
        std::vector<double> g2 = {4.5, 4.7, 4.3, 4.6, 4.4};
        auto r = mann_whitney_u_test(g1.begin(), g1.end(), g2.begin(), g2.end());
        check("mann_whitney_u: U", r.statistic, 25.0);
        check("mann_whitney_u: p", r.p_value, 0.0088158582, 0.01);
    }

    // Kruskal-Wallis: R: kruskal.test()
    {
        std::vector<std::vector<double>> groups = {
            {6.5, 6.1, 5.9, 6.3, 6.2},
            {5.8, 5.5, 5.7, 5.6, 5.4},
            {5.1, 5.3, 5.0, 5.2, 4.9}
        };
        auto r = kruskal_wallis_test(groups);
        check("kruskal_wallis: H", r.statistic, 12.5);
        check("kruskal_wallis: df", r.df, 2.0);
        check("kruskal_wallis: p", r.p_value, 0.0019304541);
    }

    // friedman_test: not implemented in statcpp

    // Shapiro-Wilk: R: shapiro.test()
    {
        std::vector<double> x = {0.11, 0.84, 0.53, 0.92, 0.32, 0.76, 0.45, 0.67, 0.28, 0.95};
        auto r = shapiro_wilk_test(x.begin(), x.end());
        check("shapiro_wilk: W", r.statistic, 0.947279275479796, 1e-3);
        check("shapiro_wilk: p", r.p_value, 0.636436237100185, 0.05);
    }

    // Levene test: R: median-based (manual ANOVA on abs deviations)
    {
        std::vector<std::vector<double>> groups = {
            {6.5, 6.1, 5.9, 6.3, 6.2},
            {5.8, 5.5, 5.7, 5.6, 5.4},
            {5.1, 5.3, 5.0, 5.2, 4.9}
        };
        auto r = levene_test(groups);
        check("levene: F", r.statistic, 0.250000000, 1e-4);
        check("levene: p", r.p_value, 0.782757789696, 1e-4);
    }

    // Bartlett test: R: bartlett.test()
    {
        std::vector<std::vector<double>> groups = {
            {6.5, 6.1, 5.9, 6.3, 6.2},
            {5.8, 5.5, 5.7, 5.6, 5.4},
            {5.1, 5.3, 5.0, 5.2, 4.9}
        };
        auto r = bartlett_test(groups);
        check("bartlett: K2", r.statistic, 0.611636532463432, 1e-4);
        check("bartlett: p", r.p_value, 0.736520457932485, 1e-4);
    }

    // Fisher exact test: R: fisher.test(matrix(c(10,3,5,12),nrow=2))
    {
        auto r = fisher_exact_test(10, 5, 3, 12);
        check("fisher_exact: p", r.p_value, 0.0253276870336761, 1e-3);
    }

    // =========================================================================
    section("ANOVA");
    // =========================================================================

    // One-way ANOVA: R: summary(aov(vals ~ grp))
    {
        std::vector<std::vector<double>> groups = {
            {6.5, 6.1, 5.9, 6.3, 6.2},
            {5.8, 5.5, 5.7, 5.6, 5.4},
            {5.1, 5.3, 5.0, 5.2, 4.9}
        };
        auto r = one_way_anova(groups);
        check("anova: F", r.between.f_statistic, 45.5);
        check("anova: p", r.between.p_value, 2.50071459070926e-06, 1e-8);
        check("anova: MSb", r.between.ms, 1.51666666666667);
        check("anova: MSw", r.within.ms, 0.0333333333333333);

        // Effect sizes from ANOVA
        check("eta_squared(anova)", statcpp::eta_squared(r), 0.883495145631068, 1e-6);
        check("omega_squared(anova)", statcpp::omega_squared(r), 0.855769230769231, 1e-6);
        check("cohens_f(anova)", statcpp::cohens_f(r), 2.75378527364305, 1e-4);

        // Tukey HSD: R: TukeyHSD(aov(...))
        // Note: C++ uses group_i - group_j (0-1), R uses group_j - group_i (2-1)
        // So C++ diff has opposite sign from R.  |diff| and p-values match.
        auto tukey = tukey_hsd(r, groups);
        check("tukey 0-1: |diff|", std::abs(tukey.comparisons[0].mean_diff), 0.6, 1e-6);
        check("tukey 0-1: p", tukey.comparisons[0].p_value, 6.043169e-04, 1e-4);
        check("tukey 0-1: CI width", tukey.comparisons[0].upper - tukey.comparisons[0].lower,
              0.9080584 - 0.2919416, 1e-3);
        check("tukey 0-2: |diff|", std::abs(tukey.comparisons[1].mean_diff), 1.1, 1e-6);
        check("tukey 0-2: p", tukey.comparisons[1].p_value, 1.683961e-06, 1e-6);
        check("tukey 1-2: |diff|", std::abs(tukey.comparisons[2].mean_diff), 0.5, 1e-6);
        check("tukey 1-2: p", tukey.comparisons[2].p_value, 2.599763e-03, 1e-3);
    }

    // =========================================================================
    section("LINEAR REGRESSION");
    // =========================================================================

    // Simple regression: R: lm(y ~ x)
    {
        std::vector<double> x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        std::vector<double> y = {2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 13.9, 16.1, 17.8, 20.2};
        auto r = simple_linear_regression(x.begin(), x.end(), y.begin(), y.end());
        check("simple_reg: intercept", r.intercept, 0.0133333333333324, 1e-6);
        check("simple_reg: slope", r.slope, 1.99939393939394, 1e-6);
        check("simple_reg: R2", r.r_squared, 0.999366775763783, 1e-6);
        check("simple_reg: se_b0", r.intercept_se, 0.110407875579141, 1e-4);
        check("simple_reg: se_b1", r.slope_se, 0.0177938403101857, 1e-4);
    }

    // Multiple regression: R: lm(Y ~ X1 + X2)
    {
        std::vector<std::vector<double>> X = {
            {1, 5}, {2, 4}, {3, 7}, {4, 3}, {5, 8},
            {6, 2}, {7, 9}, {8, 1}, {9, 6}, {10, 10}
        };
        std::vector<double> y = {3.1, 4.9, 8.2, 6.8, 11.1, 8.0, 14.9, 9.1, 14.8, 19.2};
        auto r = multiple_linear_regression(X, y);
        check("multi_reg: b0", r.coefficients[0], -1.26233009708738, 1e-4);
        check("multi_reg: b1", r.coefficients[1], 1.25701534606953, 1e-4);
        check("multi_reg: b2", r.coefficients[2], 0.792499217037269, 1e-4);
        check("multi_reg: R2", r.r_squared, 0.992021201971393, 1e-4);
    }

    // =========================================================================
    section("GLM (Logistic & Poisson)");
    // =========================================================================

    // Logistic regression: R: glm(y ~ x, family=binomial)
    {
        std::vector<std::vector<double>> X = {{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}};
        std::vector<double> y = {0,0,0,1,0,1,1,1,1,1};
        auto r = logistic_regression(X, y);
        check("logistic: intercept", r.coefficients[0], -5.8246007057683, 0.5);
        check("logistic: slope", r.coefficients[1], 1.29543707708647, 0.1);
        check("logistic: deviance", r.residual_deviance, 5.01379227698863, 0.5);
        check("logistic: null_deviance", r.null_deviance, 13.4602333401851, 0.5);
        check("logistic: AIC", r.aic, 9.01379227698863, 0.5);
    }

    // Poisson regression: R: glm(y ~ x, family=poisson)
    {
        std::vector<std::vector<double>> X = {{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}};
        std::vector<double> y = {0,1,2,3,5,4,7,6,8,10};
        auto r = poisson_regression(X, y);
        check("poisson: intercept", r.coefficients[0], -0.0639958806670219, 0.1);
        check("poisson: slope", r.coefficients[1], 0.245869417195599, 0.05);
        check("poisson: deviance", r.residual_deviance, 4.29551375075109, 0.5);
        check("poisson: AIC", r.aic, 38.2076555825617, 1.0);
    }

    // =========================================================================
    section("EFFECT SIZE");
    // =========================================================================

    // Cohen's d (two-sample): R manual calculation
    {
        std::vector<double> g1 = {5.1, 4.9, 5.3, 5.1, 4.8};
        std::vector<double> g2 = {4.5, 4.7, 4.3, 4.6, 4.4};
        double d = cohens_d_two_sample(g1.begin(), g1.end(), g2.begin(), g2.end());
        check("cohens_d_two_sample", d, 3.04255531702266, 1e-4);
    }

    // Hedges' g (two-sample)
    {
        std::vector<double> g1 = {5.1, 4.9, 5.3, 5.1, 4.8};
        std::vector<double> g2 = {4.5, 4.7, 4.3, 4.6, 4.4};
        double g = hedges_g_two_sample(g1.begin(), g1.end(), g2.begin(), g2.end());
        check("hedges_g_two_sample", g, 2.74811447989144, 1e-4);
    }

    // eta_squared (standalone)
    {
        double eta2 = eta_squared(2.5, 5.0);
        check("eta_squared(2.5, 5.0)", eta2, 0.5);
    }

    // =========================================================================
    section("ESTIMATION (CI)");
    // =========================================================================

    // CI for mean: R: t.test(x, conf.level=0.95)
    {
        std::vector<double> x = {23.1, 25.4, 22.8, 24.5, 23.9, 25.1, 24.0, 23.5, 24.8, 25.0};
        auto ci = ci_mean(x.begin(), x.end(), 0.95);
        check("ci_mean: lower", ci.lower, 23.5733278883029, 1e-4);
        check("ci_mean: upper", ci.upper, 24.8466721116971, 1e-4);
    }

    // =========================================================================
    section("CATEGORICAL");
    // =========================================================================

    // Odds ratio: [[30,10],[20,40]] -> OR = (30*40)/(10*20) = 6.0
    {
        std::size_t a=30, b=10, c=20, d=40;
        auto r = odds_ratio(a, b, c, d);
        check("odds_ratio", r.odds_ratio, 6.0);
        check("odds_ratio: CI lower", r.ci_lower, 2.45259331002879, 1e-3);
        check("odds_ratio: CI upper", r.ci_upper, 14.678340617172, 1e-2);
    }

    // Relative risk: [[30,10],[20,40]] -> RR = (30/40)/(20/60) = 2.25
    {
        std::size_t a=30, b=10, c=20, d=40;
        auto r = relative_risk(a, b, c, d);
        check("relative_risk", r.relative_risk, 2.25);
        check("relative_risk: CI lower", r.ci_lower, 1.50809443689666, 1e-3);
        check("relative_risk: CI upper", r.ci_upper, 3.35688526934531, 1e-2);
    }

    // Risk difference: p1-p2 = 30/40 - 20/60 = 0.75 - 0.3333 = 0.4167
    {
        std::size_t a=30, b=10, c=20, d=40;
        auto r = risk_difference(a, b, c, d);
        check("risk_difference", r.risk_difference, 0.416666666666667, 1e-6);
        check("risk_difference: CI lower", r.ci_lower, 0.237123780012396, 1e-3);
        check("risk_difference: CI upper", r.ci_upper, 0.596209553320937, 1e-3);
    }

    // NNT = 1/RD = 2.4
    {
        auto nnt = number_needed_to_treat(
            std::vector<std::vector<std::size_t>>{{30, 10}, {20, 40}});
        check("NNT", nnt, 2.4);
    }

    // =========================================================================
    section("SHAPE OF DISTRIBUTION");
    // =========================================================================

    // R e1071::skewness type 1 = population, type 2 = sample
    {
        std::vector<double> s = {2.0, 4.0, 4.5, 4.7, 5.0, 5.0, 5.1, 5.5, 6.0, 8.0};
        double ps = population_skewness(s.begin(), s.end());
        double ss = sample_skewness(s.begin(), s.end());
        check("population_skewness", ps, 0.0409311186796388, 1e-4);
        check("sample_skewness", ss, 0.048538335827371, 1e-4);

        double pk = population_kurtosis(s.begin(), s.end());
        check("population_kurtosis (excess)", pk, 0.870400914658122, 1e-4);
    }

    // =========================================================================
    section("SURVIVAL");
    // =========================================================================

    // Kaplan-Meier: R: survfit(Surv(times, status) ~ 1)
    {
        std::vector<double> times = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        std::vector<bool> events = {true, false, true, true, false, true, false, true, true, false};
        auto km = kaplan_meier(times, events);

        // R event times: 1,3,4,6,8,9  surv: 0.9,0.7875,0.675,0.54,0.36,0.18
        check("km: S(0)", km.survival[0], 1.0);
        check("km: S(1)", km.survival[1], 0.9);
        check("km: S(3)", km.survival[2], 0.7875);
        check("km: S(4)", km.survival[3], 0.675);
        check("km: S(6)", km.survival[4], 0.54);
        check("km: S(8)", km.survival[5], 0.36);
        check("km: S(9)", km.survival[6], 0.18);

        // Standard errors
        check("km: se(1)", km.se[1], 0.09486832981, 1e-3);
        check("km: se(3)", km.se[2], 0.13403299500, 1e-3);
        check("km: se(4)", km.se[3], 0.15507054846, 1e-3);

        // Median survival time
        double med = median_survival_time(km);
        check("km: median survival", med, 8.0);
    }

    // Nelson-Aalen estimator
    {
        std::vector<double> times = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        std::vector<bool> events = {true, false, true, true, false, true, false, true, true, false};
        auto na = nelson_aalen(times, events);

        check("na: H(1)", na.cumulative_hazard[1], 0.1);
        check("na: H(3)", na.cumulative_hazard[2], 0.225);
        check("na: H(4)", na.cumulative_hazard[3], 0.3678571429, 1e-4);
        check("na: H(6)", na.cumulative_hazard[4], 0.5678571429, 1e-4);
        check("na: H(8)", na.cumulative_hazard[5], 0.9011904762, 1e-4);
        check("na: H(9)", na.cumulative_hazard[6], 1.4011904762, 1e-4);
    }

    // Log-rank test: R: survdiff(Surv(times, status) ~ group)
    {
        std::vector<double> times1 = {1, 3, 5, 7, 9, 11, 13, 15};
        std::vector<bool> events1 = {true, true, false, true, true, false, true, true};
        std::vector<double> times2 = {2, 4, 6, 8, 10, 12, 14, 16};
        std::vector<bool> events2 = {true, false, true, true, false, true, true, false};
        auto r = logrank_test(times1, events1, times2, events2);
        check("logrank: chisq", r.statistic, 0.300011688806271, 0.1);
        check("logrank: p", r.p_value, 0.583875093032456, 0.1);
        check("logrank: obs1", static_cast<double>(r.observed1), 6.0);
        check("logrank: obs2", static_cast<double>(r.observed2), 5.0);
        check("logrank: exp1", r.expected1, 5.098989899, 0.5);
        check("logrank: exp2", r.expected2, 5.901010101, 0.5);
    }

    // =========================================================================
    section("MODEL SELECTION");
    // =========================================================================

    // AIC: R: -2*(-50) + 2*3 = 106
    {
        double aic_val = aic(-50.0, 3);
        check("AIC(-50, k=3)", aic_val, 106.0);
    }

    // BIC: R: -2*(-50) + 3*log(100) = 100 + 13.8155 = 113.8155
    {
        double bic_val = bic(-50.0, 100, 3);
        check("BIC(-50, k=3, n=100)", bic_val, 100.0 + 3 * std::log(100.0));
    }

    // AICc: R: AIC + 2*k*(k+1)/(n-k-1) = 106 + 24/96 = 106.25
    {
        double aicc_val = aicc(-50.0, 100, 3);
        check("AICc(-50, k=3, n=100)", aicc_val, 106.0 + 2.0 * 3.0 * 4.0 / 96.0);
    }

    // =========================================================================
    section("POWER ANALYSIS");
    // =========================================================================

    // Power analysis values (pwr package not available, use range checks)
    {
        double pw = power_t_test_one_sample(0.5, 30);
        printf("  power_t_test_one_sample(d=0.5, n=30) = %.6f\n", pw);
        check("power_one_sample: reasonable", (pw > 0.5 && pw < 1.0) ? 1.0 : 0.0, 1.0);
    }
    {
        double pw2 = power_t_test_two_sample(0.5, 30, 30);
        printf("  power_t_test_two_sample(d=0.5, n1=30, n2=30) = %.6f\n", pw2);
        check("power_two_sample: reasonable", (pw2 > 0.3 && pw2 < 1.0) ? 1.0 : 0.0, 1.0);
    }
    {
        auto n = sample_size_t_test_one_sample(0.5, 0.80);
        printf("  sample_size_t_test_one_sample(d=0.5, power=0.8) = %zu\n", n);
        check("ss_one_sample: reasonable", (n >= 25 && n <= 45) ? 1.0 : 0.0, 1.0);
    }
    {
        auto n = sample_size_t_test_two_sample(0.5, 0.80);
        printf("  sample_size_t_test_two_sample(d=0.5, power=0.8) = %zu\n", n);
        check("ss_two_sample: reasonable", (n >= 50 && n <= 80) ? 1.0 : 0.0, 1.0);
    }

    // =========================================================================
    section("ROBUST STATISTICS");
    // =========================================================================

    // MAD: R: mad(x, constant=1) = 0.6
    {
        std::vector<double> x = {1.2, 2.3, 2.5, 2.8, 3.1, 3.3, 3.5, 3.7, 4.0, 15.0};
        double m = mad(x.begin(), x.end());
        check("mad (unscaled)", m, 0.6, 1e-6);
    }

    // MAD scaled: R: mad(x) = 0.88956
    {
        std::vector<double> x = {1.2, 2.3, 2.5, 2.8, 3.1, 3.3, 3.5, 3.7, 4.0, 15.0};
        double m = mad_scaled(x.begin(), x.end());
        check("mad_scaled", m, 0.88956, 1e-3);
    }

    // Winsorize: R manual at 10%
    {
        std::vector<double> x = {1.2, 2.3, 2.5, 2.8, 3.1, 3.3, 3.5, 3.7, 4.0, 15.0};
        auto w = winsorize(x.begin(), x.end(), 0.10);
        std::vector<double> ws(w.begin(), w.end());
        std::sort(ws.begin(), ws.end());
        // R quantile-based: lo=quantile(0.10)=2.19, hi=quantile(0.90)=5.1
        // sorted result: 2.19, 2.30, 2.50, 2.80, 3.10, 3.30, 3.50, 3.70, 4.00, 5.10
        check("winsorize[0]", ws[0], 2.19, 1e-4);
        check("winsorize[1]", ws[1], 2.30, 1e-4);
        check("winsorize[8]", ws[8], 4.0);
        check("winsorize[9]", ws[9], 5.1, 1e-4);
    }

    // =========================================================================
    section("TIME SERIES");
    // =========================================================================

    // ACF: R: acf(y, lag.max=3)
    {
        std::vector<double> y = {1.0, 2.0, 3.0, 2.5, 4.0, 3.5, 5.0, 4.5, 6.0, 5.5};
        auto acf_vals = acf(y.begin(), y.end(), 3);
        // acf_vals[0] = lag 0 = 1.0 (if included), or lag 1
        // Check the C++ API: acf returns vector for lag 0..max_lag
        check("acf lag0", acf_vals[0], 1.0);
        check("acf lag1", acf_vals[1], 0.561038961038961, 1e-4);
        check("acf lag2", acf_vals[2], 0.373160173160173, 1e-4);
        check("acf lag3", acf_vals[3], 0.148484848484848, 1e-4);
    }

    // PACF: R: pacf(y, lag.max=3)
    {
        std::vector<double> y = {1.0, 2.0, 3.0, 2.5, 4.0, 3.5, 5.0, 4.5, 6.0, 5.5};
        auto pacf_vals = pacf(y.begin(), y.end(), 3);
        // C++ PACF: [0]=1.0 (lag 0), [1]=lag 1, etc.  R pacf() starts at lag 1.
        check("pacf lag0", pacf_vals[0], 1.0);
        check("pacf lag1", pacf_vals[1], 0.561038961038961, 1e-3);
        check("pacf lag2", pacf_vals[2], 0.0852195715884439, 1e-2);
        check("pacf lag3", pacf_vals[3], -0.133541371654217, 0.05);
    }

    // diff: R: diff(y)
    {
        std::vector<double> y = {1.0, 2.0, 3.0, 2.5, 4.0, 3.5, 5.0, 4.5, 6.0, 5.5};
        auto d = diff(y.begin(), y.end(), 1);
        check("diff[0]", d[0], 1.0);
        check("diff[1]", d[1], 1.0);
        check("diff[2]", d[2], -0.5);
        check("diff[3]", d[3], 1.5);
    }

    // Forecast error metrics: MAE, MSE, RMSE
    {
        std::vector<double> actual = {3, -0.5, 2, 7};
        std::vector<double> predicted = {2.5, 0.0, 2, 8};
        double mae_val = mae(actual.begin(), actual.end(), predicted.begin());
        double mse_val = mse(actual.begin(), actual.end(), predicted.begin());
        double rmse_val = rmse(actual.begin(), actual.end(), predicted.begin());
        check("MAE", mae_val, 0.5);
        check("MSE", mse_val, 0.375);
        check("RMSE", rmse_val, 0.612372435695794, 1e-6);
    }

    // =========================================================================
    section("MULTIVARIATE");
    // =========================================================================

    // Covariance matrix: R: cov(X)
    {
        std::vector<std::vector<double>> X = {
            {1, 5, 3}, {2, 4, 6}, {3, 7, 2}, {4, 3, 8}
        };
        auto cv = covariance_matrix(X);
        check("cov[0][0]", cv[0][0], 1.66666666666667, 1e-4);
        check("cov[0][1]", cv[0][1], -0.5, 1e-4);
        check("cov[0][2]", cv[0][2], 1.83333333333333, 1e-4);
        check("cov[1][1]", cv[1][1], 2.91666666666667, 1e-4);
        check("cov[1][2]", cv[1][2], -4.41666666666667, 1e-4);
        check("cov[2][2]", cv[2][2], 7.58333333333333, 1e-4);
    }

    // Correlation matrix: R: cor(X)
    {
        std::vector<std::vector<double>> X = {
            {1, 5, 3}, {2, 4, 6}, {3, 7, 2}, {4, 3, 8}
        };
        auto cr = correlation_matrix(X);
        check("cor[0][0]", cr[0][0], 1.0);
        check("cor[0][1]", cr[0][1], -0.226778683805536, 1e-4);
        check("cor[0][2]", cr[0][2], 0.515687954032345, 1e-4);
        check("cor[1][2]", cr[1][2], -0.939120133318293, 1e-4);
    }

    // =========================================================================
    section("DATA WRANGLING");
    // =========================================================================

    // log_transform: R: log(x)
    {
        std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
        auto lt = log_transform(x);
        check("log_transform[0]", lt[0], 0.0);
        check("log_transform[1]", lt[1], std::log(2.0));
        check("log_transform[4]", lt[4], std::log(5.0));
    }

    // sqrt_transform: R: sqrt(x)
    {
        std::vector<double> x = {1.0, 4.0, 9.0, 16.0, 25.0};
        auto st = sqrt_transform(x);
        check("sqrt_transform[0]", st[0], 1.0);
        check("sqrt_transform[1]", st[1], 2.0);
        check("sqrt_transform[4]", st[4], 5.0);
    }

    // =========================================================================
    // FINAL SUMMARY
    // =========================================================================
    if (g_section_fail > 0) {
        printf("  >> %d FAILURES in %s\n", g_section_fail, g_section);
    }
    printf("\n========================================\n");
    printf("TOTAL: %d PASS, %d FAIL\n", g_pass, g_fail);
    printf("========================================\n");

    return g_fail > 0 ? 1 : 0;
}
