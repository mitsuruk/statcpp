#!/usr/bin/env Rscript
#
# generate_reference_values.R
#
# Generates all reference values used in verify_vs_r.cpp.
# Each section matches the corresponding section in verify_vs_r.cpp.
#
# Usage:
#   Rscript generate_reference_values.R
#
# Requirements:
#   - R >= 4.4.2
#   - Packages: e1071, survival (install with install.packages())
#

fmt <- function(x) sprintf("%.15g", x)

cat("=== PARAMETRIC TESTS ===\n\n")

# One-sample t-test
x <- c(5.1, 4.9, 5.3, 5.1, 4.8, 5.2, 5.0, 4.9)
r <- t.test(x, mu = 5.0)
cat("t_test(one-sample): t  =", fmt(r$statistic), "\n")
cat("t_test(one-sample): df =", fmt(r$parameter), "\n")
cat("t_test(one-sample): p  =", fmt(r$p.value), "\n\n")

# Two-sample t-test (equal variance)
g1 <- c(5.1, 4.9, 5.3, 5.1, 4.8)
g2 <- c(4.5, 4.7, 4.3, 4.6, 4.4)
r <- t.test(g1, g2, var.equal = TRUE)
cat("t_test_two_sample: t  =", fmt(r$statistic), "\n")
cat("t_test_two_sample: df =", fmt(r$parameter), "\n")
cat("t_test_two_sample: p  =", fmt(r$p.value), "\n\n")

# Welch t-test
r <- t.test(g1, g2)
cat("t_test_welch: t  =", fmt(r$statistic), "\n")
cat("t_test_welch: df =", fmt(r$parameter), "\n")
cat("t_test_welch: p  =", fmt(r$p.value), "\n\n")

# Paired t-test
pre  <- c(5.1, 4.9, 5.3, 5.1, 4.8, 5.2)
post <- c(5.5, 5.2, 5.6, 5.3, 5.1, 5.4)
r <- t.test(pre, post, paired = TRUE)
cat("t_test_paired: t  =", fmt(r$statistic), "\n")
cat("t_test_paired: df =", fmt(r$parameter), "\n")
cat("t_test_paired: p  =", fmt(r$p.value), "\n\n")

# F-test
r <- var.test(g1, g2)
cat("f_test: F =", fmt(r$statistic), "\n")
cat("f_test: p =", fmt(r$p.value), "\n\n")

# Chi-squared GOF
obs <- c(16, 18, 16, 14, 12, 12)
r <- chisq.test(obs, p = rep(1/6, 6))
cat("chisq_gof: X2 =", fmt(r$statistic), "\n")
cat("chisq_gof: df =", fmt(r$parameter), "\n")
cat("chisq_gof: p  =", fmt(r$p.value), "\n\n")

# Chi-squared independence (correct=FALSE to match C++)
m <- matrix(c(10, 30, 20, 40), nrow = 2)
r <- chisq.test(m, correct = FALSE)
cat("chisq_independence: X2 =", fmt(r$statistic), "\n")
cat("chisq_independence: df =", fmt(r$parameter), "\n")
cat("chisq_independence: p  =", fmt(r$p.value), "\n\n")

# Z-test for proportion (prop.test returns X2 = z^2)
r <- prop.test(45, 200, p = 0.20, correct = FALSE)
cat("z_test_prop: X2 (=z^2) =", fmt(r$statistic), "\n")
cat("z_test_prop: p          =", fmt(r$p.value), "\n\n")

# Two-sample proportion test
r <- prop.test(c(45, 55), c(200, 250), correct = FALSE)
cat("z_test_prop_two: X2 (=z^2) =", fmt(r$statistic), "\n")
cat("z_test_prop_two: p          =", fmt(r$p.value), "\n\n")

# Multiple testing corrections
pv <- c(0.01, 0.04, 0.03, 0.005, 0.15, 0.08)
bonf <- p.adjust(pv, method = "bonferroni")
bh   <- p.adjust(pv, method = "BH")
cat("bonferroni:", paste(fmt(bonf), collapse = ", "), "\n")
cat("BH:        ", paste(fmt(bh),   collapse = ", "), "\n\n")


cat("=== NONPARAMETRIC TESTS ===\n\n")

# Wilcoxon signed-rank (correct=TRUE is default)
x <- c(5.1, 4.9, 5.3, 5.1, 4.8, 5.2, 5.0, 4.9)
r <- wilcox.test(x, mu = 5.0)
cat("wilcoxon_signed_rank: V =", fmt(r$statistic), "\n")
cat("wilcoxon_signed_rank: p =", fmt(r$p.value), "\n\n")

# Mann-Whitney U
g1 <- c(5.1, 4.9, 5.3, 5.1, 4.8)
g2 <- c(4.5, 4.7, 4.3, 4.6, 4.4)
r <- wilcox.test(g1, g2)
cat("mann_whitney_u: W =", fmt(r$statistic), "\n")
cat("mann_whitney_u: p =", fmt(r$p.value), "\n\n")

# Kruskal-Wallis
groups_vals <- c(6.5, 6.1, 5.9, 6.3, 6.2, 5.8, 5.5, 5.7, 5.6, 5.4, 5.1, 5.3, 5.0, 5.2, 4.9)
groups_grp  <- factor(rep(1:3, each = 5))
r <- kruskal.test(groups_vals ~ groups_grp)
cat("kruskal_wallis: H  =", fmt(r$statistic), "\n")
cat("kruskal_wallis: df =", fmt(r$parameter), "\n")
cat("kruskal_wallis: p  =", fmt(r$p.value), "\n\n")

# Shapiro-Wilk
x <- c(0.11, 0.84, 0.53, 0.92, 0.32, 0.76, 0.45, 0.67, 0.28, 0.95)
r <- shapiro.test(x)
cat("shapiro_wilk: W =", fmt(r$statistic), "\n")
cat("shapiro_wilk: p =", fmt(r$p.value), "\n\n")

# Levene test (median-based, manual)
g1_lev <- c(6.5, 6.1, 5.9, 6.3, 6.2)
g2_lev <- c(5.8, 5.5, 5.7, 5.6, 5.4)
g3_lev <- c(5.1, 5.3, 5.0, 5.2, 4.9)
d1 <- abs(g1_lev - median(g1_lev))
d2 <- abs(g2_lev - median(g2_lev))
d3 <- abs(g3_lev - median(g3_lev))
lev_vals <- c(d1, d2, d3)
lev_grp  <- factor(rep(1:3, each = 5))
r <- summary(aov(lev_vals ~ lev_grp))
cat("levene: F =", fmt(r[[1]]$`F value`[1]), "\n")
cat("levene: p =", fmt(r[[1]]$`Pr(>F)`[1]), "\n\n")

# Bartlett test
r <- bartlett.test(list(g1_lev, g2_lev, g3_lev))
cat("bartlett: K2 =", fmt(r$statistic), "\n")
cat("bartlett: p  =", fmt(r$p.value), "\n\n")

# Fisher exact test
# C++ call: fisher_exact_test(10, 5, 3, 12)
# This creates matrix: [[10, 5], [3, 12]]
m <- matrix(c(10, 3, 5, 12), nrow = 2)
r <- fisher.test(m)
cat("fisher_exact: p =", fmt(r$p.value), "\n\n")


cat("=== ANOVA ===\n\n")

# One-way ANOVA
vals <- c(6.5, 6.1, 5.9, 6.3, 6.2, 5.8, 5.5, 5.7, 5.6, 5.4, 5.1, 5.3, 5.0, 5.2, 4.9)
grp  <- factor(rep(1:3, each = 5))
r <- summary(aov(vals ~ grp))
cat("anova: F   =", fmt(r[[1]]$`F value`[1]), "\n")
cat("anova: p   =", fmt(r[[1]]$`Pr(>F)`[1]), "\n")
cat("anova: MSb =", fmt(r[[1]]$`Mean Sq`[1]), "\n")
cat("anova: MSw =", fmt(r[[1]]$`Mean Sq`[2]), "\n\n")

# Effect sizes
ss_b <- r[[1]]$`Sum Sq`[1]
ss_w <- r[[1]]$`Sum Sq`[2]
ss_t <- ss_b + ss_w
df_b <- r[[1]]$Df[1]
ms_w <- r[[1]]$`Mean Sq`[2]
f_val <- r[[1]]$`F value`[1]
n <- length(vals)

eta2  <- ss_b / ss_t
omega2 <- (ss_b - df_b * ms_w) / (ss_t + ms_w)
cohens_f <- sqrt(eta2 / (1 - eta2))
cat("eta_squared   =", fmt(eta2), "\n")
cat("omega_squared =", fmt(omega2), "\n")
cat("cohens_f      =", fmt(cohens_f), "\n\n")

# Tukey HSD
fit <- aov(vals ~ grp)
tukey <- TukeyHSD(fit)
cat("Tukey HSD:\n")
print(tukey)
cat("\n")


cat("=== LINEAR REGRESSION ===\n\n")

# Simple regression
x <- 1:10
y <- c(2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 13.9, 16.1, 17.8, 20.2)
r <- summary(lm(y ~ x))
cat("simple_reg: intercept =", fmt(r$coefficients[1, 1]), "\n")
cat("simple_reg: slope     =", fmt(r$coefficients[2, 1]), "\n")
cat("simple_reg: R2        =", fmt(r$r.squared), "\n")
cat("simple_reg: se_b0     =", fmt(r$coefficients[1, 2]), "\n")
cat("simple_reg: se_b1     =", fmt(r$coefficients[2, 2]), "\n\n")

# Multiple regression
X1 <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
X2 <- c(5, 4, 7, 3, 8, 2, 9, 1, 6, 10)
Y  <- c(3.1, 4.9, 8.2, 6.8, 11.1, 8.0, 14.9, 9.1, 14.8, 19.2)
r <- summary(lm(Y ~ X1 + X2))
cat("multi_reg: b0 =", fmt(r$coefficients[1, 1]), "\n")
cat("multi_reg: b1 =", fmt(r$coefficients[2, 1]), "\n")
cat("multi_reg: b2 =", fmt(r$coefficients[3, 1]), "\n")
cat("multi_reg: R2 =", fmt(r$r.squared), "\n\n")


cat("=== GLM (Logistic & Poisson) ===\n\n")

# Logistic regression
x_glm <- 1:10
y_glm <- c(0, 0, 0, 1, 0, 1, 1, 1, 1, 1)
r <- glm(y_glm ~ x_glm, family = binomial)
cat("logistic: intercept     =", fmt(coef(r)[1]), "\n")
cat("logistic: slope         =", fmt(coef(r)[2]), "\n")
cat("logistic: deviance      =", fmt(r$deviance), "\n")
cat("logistic: null_deviance =", fmt(r$null.deviance), "\n")
cat("logistic: AIC           =", fmt(r$aic), "\n\n")

# Poisson regression
y_pois <- c(0, 1, 2, 3, 5, 4, 7, 6, 8, 10)
r <- glm(y_pois ~ x_glm, family = poisson)
cat("poisson: intercept =", fmt(coef(r)[1]), "\n")
cat("poisson: slope     =", fmt(coef(r)[2]), "\n")
cat("poisson: deviance  =", fmt(r$deviance), "\n")
cat("poisson: AIC       =", fmt(r$aic), "\n\n")


cat("=== EFFECT SIZE ===\n\n")

# Cohen's d (two-sample, pooled SD)
g1 <- c(5.1, 4.9, 5.3, 5.1, 4.8)
g2 <- c(4.5, 4.7, 4.3, 4.6, 4.4)
n1 <- length(g1); n2 <- length(g2)
sp <- sqrt(((n1-1)*var(g1) + (n2-1)*var(g2)) / (n1+n2-2))
d <- (mean(g1) - mean(g2)) / sp
cat("cohens_d_two_sample =", fmt(d), "\n")

# Hedges' g (with J correction)
J <- 1 - 3 / (4*(n1+n2-2) - 1)
g_val <- d * J
cat("hedges_g_two_sample =", fmt(g_val), "\n")

# eta_squared standalone
cat("eta_squared(2.5, 5.0) =", fmt(2.5 / (2.5 + 5.0)), "\n\n")


cat("=== ESTIMATION (CI) ===\n\n")

x <- c(23.1, 25.4, 22.8, 24.5, 23.9, 25.1, 24.0, 23.5, 24.8, 25.0)
r <- t.test(x, conf.level = 0.95)
cat("ci_mean: lower =", fmt(r$conf.int[1]), "\n")
cat("ci_mean: upper =", fmt(r$conf.int[2]), "\n\n")


cat("=== CATEGORICAL ===\n\n")

# 2x2 table: [[30, 10], [20, 40]]
# a=30, b=10, c=20, d=40
a <- 30; b <- 10; c <- 20; d <- 40

# Odds ratio
OR <- (a * d) / (b * c)
se_log_or <- sqrt(1/a + 1/b + 1/c + 1/d)
cat("odds_ratio          =", fmt(OR), "\n")
cat("odds_ratio: CI lower =", fmt(exp(log(OR) - 1.96 * se_log_or)), "\n")
cat("odds_ratio: CI upper =", fmt(exp(log(OR) + 1.96 * se_log_or)), "\n\n")

# Relative risk
# p1 = a/(a+b) = 30/40 = 0.75, p2 = c/(c+d) = 20/60 = 1/3
p1 <- a / (a + b)
p2 <- c / (c + d)
RR <- p1 / p2
se_log_rr <- sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
cat("relative_risk          =", fmt(RR), "\n")
cat("relative_risk: CI lower =", fmt(exp(log(RR) - 1.96 * se_log_rr)), "\n")
cat("relative_risk: CI upper =", fmt(exp(log(RR) + 1.96 * se_log_rr)), "\n\n")

# Risk difference
RD <- p1 - p2
se_rd <- sqrt(p1*(1-p1)/(a+b) + p2*(1-p2)/(c+d))
cat("risk_difference          =", fmt(RD), "\n")
cat("risk_difference: CI lower =", fmt(RD - 1.96 * se_rd), "\n")
cat("risk_difference: CI upper =", fmt(RD + 1.96 * se_rd), "\n\n")

# NNT
cat("NNT =", fmt(1 / RD), "\n\n")


cat("=== SHAPE OF DISTRIBUTION ===\n\n")

# Requires e1071 for skewness/kurtosis types
if (requireNamespace("e1071", quietly = TRUE)) {
  s <- c(2.0, 4.0, 4.5, 4.7, 5.0, 5.0, 5.1, 5.5, 6.0, 8.0)
  cat("population_skewness (type 1) =", fmt(e1071::skewness(s, type = 1)), "\n")
  cat("sample_skewness (type 2)     =", fmt(e1071::skewness(s, type = 2)), "\n")
  cat("population_kurtosis (type 1) =", fmt(e1071::kurtosis(s, type = 1)), "\n\n")
} else {
  cat("SKIP: e1071 package not installed\n\n")
}


cat("=== SURVIVAL ===\n\n")

if (requireNamespace("survival", quietly = TRUE)) {
  library(survival)

  # Kaplan-Meier
  times  <- 1:10
  status <- c(1, 0, 1, 1, 0, 1, 0, 1, 1, 0)
  fit <- survfit(Surv(times, status) ~ 1)
  cat("Kaplan-Meier:\n")
  print(summary(fit))
  cat("\n")
  cat("km: median survival =", fmt(fit$table["median"]), "\n\n")

  # Nelson-Aalen cumulative hazard (manual: sum of d_i / n_i at event times)
  cat("Nelson-Aalen cumulative hazard (manual):\n")
  event_times <- times[status == 1]
  cumhaz <- 0
  for (t in event_times) {
    n_at_risk <- sum(times >= t)
    cumhaz <- cumhaz + 1 / n_at_risk
    cat("  H(", t, ") =", fmt(cumhaz), "\n")
  }
  cat("\n")

  # Log-rank test
  times1  <- c(1, 3, 5, 7, 9, 11, 13, 15)
  events1 <- c(1, 1, 0, 1, 1, 0, 1, 1)
  times2  <- c(2, 4, 6, 8, 10, 12, 14, 16)
  events2 <- c(1, 0, 1, 1, 0, 1, 1, 0)
  all_times  <- c(times1, times2)
  all_events <- c(events1, events2)
  all_groups <- factor(c(rep(1, 8), rep(2, 8)))
  r <- survdiff(Surv(all_times, all_events) ~ all_groups)
  cat("logrank: chisq =", fmt(r$chisq), "\n")
  cat("logrank: p     =", fmt(1 - pchisq(r$chisq, 1)), "\n")
  cat("logrank: obs1  =", fmt(r$obs[1]), "\n")
  cat("logrank: obs2  =", fmt(r$obs[2]), "\n")
  cat("logrank: exp1  =", fmt(r$exp[1]), "\n")
  cat("logrank: exp2  =", fmt(r$exp[2]), "\n\n")
} else {
  cat("SKIP: survival package not installed\n\n")
}


cat("=== MODEL SELECTION ===\n\n")

logL <- -50.0; k <- 3; n <- 100
aic_val  <- -2 * logL + 2 * k
bic_val  <- -2 * logL + k * log(n)
aicc_val <- aic_val + 2 * k * (k + 1) / (n - k - 1)
cat("AIC  =", fmt(aic_val), "\n")
cat("BIC  =", fmt(bic_val), "\n")
cat("AICc =", fmt(aicc_val), "\n\n")


cat("=== POWER ANALYSIS ===\n\n")

if (requireNamespace("pwr", quietly = TRUE)) {
  library(pwr)
  r <- pwr.t.test(d = 0.5, n = 30, type = "one.sample", alternative = "two.sided")
  cat("power_one_sample(d=0.5, n=30)  =", fmt(r$power), "\n")
  r <- pwr.t.test(d = 0.5, n = 30, type = "two.sample", alternative = "two.sided")
  cat("power_two_sample(d=0.5, n=30)  =", fmt(r$power), "\n")
  r <- pwr.t.test(d = 0.5, power = 0.80, type = "one.sample", alternative = "two.sided")
  cat("ss_one_sample(d=0.5, power=0.8) = n =", ceiling(r$n), "\n")
  r <- pwr.t.test(d = 0.5, power = 0.80, type = "two.sample", alternative = "two.sided")
  cat("ss_two_sample(d=0.5, power=0.8) = n =", ceiling(r$n), "\n\n")
} else {
  cat("SKIP: pwr package not installed (range checks used in C++)\n\n")
}


cat("=== ROBUST STATISTICS ===\n\n")

x <- c(1.2, 2.3, 2.5, 2.8, 3.1, 3.3, 3.5, 3.7, 4.0, 15.0)
cat("mad (unscaled, constant=1) =", fmt(mad(x, constant = 1)), "\n")
cat("mad_scaled (default)       =", fmt(mad(x)), "\n\n")

# Winsorize at 10%
lo <- quantile(x, 0.10)
hi <- quantile(x, 0.90)
w <- pmax(pmin(x, hi), lo)
cat("winsorize 10% boundaries: lo =", fmt(lo), " hi =", fmt(hi), "\n")
cat("winsorize result (sorted):  ", paste(fmt(sort(w)), collapse = ", "), "\n\n")


cat("=== TIME SERIES ===\n\n")

y <- c(1.0, 2.0, 3.0, 2.5, 4.0, 3.5, 5.0, 4.5, 6.0, 5.5)

# ACF
a <- acf(y, lag.max = 3, plot = FALSE)
cat("acf lag0 =", fmt(a$acf[1]), "\n")
cat("acf lag1 =", fmt(a$acf[2]), "\n")
cat("acf lag2 =", fmt(a$acf[3]), "\n")
cat("acf lag3 =", fmt(a$acf[4]), "\n\n")

# PACF (R's pacf starts at lag 1)
p <- pacf(y, lag.max = 3, plot = FALSE)
cat("pacf lag1 =", fmt(p$acf[1]), "\n")
cat("pacf lag2 =", fmt(p$acf[2]), "\n")
cat("pacf lag3 =", fmt(p$acf[3]), "\n\n")

# diff
d <- diff(y)
cat("diff:", paste(fmt(d), collapse = ", "), "\n\n")

# Forecast error metrics
actual    <- c(3, -0.5, 2, 7)
predicted <- c(2.5, 0.0, 2, 8)
mae_val  <- mean(abs(actual - predicted))
mse_val  <- mean((actual - predicted)^2)
rmse_val <- sqrt(mse_val)
cat("MAE  =", fmt(mae_val), "\n")
cat("MSE  =", fmt(mse_val), "\n")
cat("RMSE =", fmt(rmse_val), "\n\n")


cat("=== MULTIVARIATE ===\n\n")

X <- matrix(c(1, 2, 3, 4,
              5, 4, 7, 3,
              3, 6, 2, 8), nrow = 4, ncol = 3)
cat("Covariance matrix:\n")
print(cov(X), digits = 15)
cat("\nCorrelation matrix:\n")
print(cor(X), digits = 15)
cat("\n")


cat("=== DATA WRANGLING ===\n\n")

x <- c(1.0, 2.0, 3.0, 4.0, 5.0)
cat("log_transform:", paste(fmt(log(x)), collapse = ", "), "\n")

x <- c(1.0, 4.0, 9.0, 16.0, 25.0)
cat("sqrt_transform:", paste(fmt(sqrt(x)), collapse = ", "), "\n\n")


cat("========================================\n")
cat("Reference value generation complete.\n")
cat("========================================\n")
