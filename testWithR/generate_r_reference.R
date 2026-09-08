#!/usr/bin/env Rscript
#
# generate_r_reference.R
#
# Emits C++ headers holding R reference values for the statcpp R-verification
# tests. R is the single source of truth for both the input data and the
# expected results: the generated header carries both, so there is no manual
# transcription step and no chance of the two drifting apart.
#
# Values are written as C++17 hexadecimal floating-point literals ("%a"), which
# round-trip exactly, so the generated constants are bit-identical to what R
# computed. Any remaining difference seen by a test is a genuine algorithmic
# difference between statcpp and R.
#
# R is never linked into the test binary. It runs here as a separate process and
# only numbers cross the boundary, so the GPL-licensed R implementation does not
# affect the MIT license of statcpp or of the test binary.
#
# Usage:
#   Rscript generate_r_reference.R
#
# Requirements:
#   R >= 4.4.2, packages: nortest, car, effectsize, e1071, proxy, performance,
#                          survival, cluster, zoo
#
# The output is deterministic: re-running it must not change the generated
# headers. Do not edit the generated headers by hand.

suppressPackageStartupMessages({
    library(nortest)
    library(car)
    library(effectsize)
    library(e1071)
    library(proxy)
    library(performance)
    library(survival)
    library(cluster)
    library(zoo)
})

R_VERSION_USED <- "4.4.2"

# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------

.env <- new.env(parent = emptyenv())
.env$cases <- list()

# Format one double as an exact C++ literal.
cxx_double <- function(x) {
    if (is.na(x) && !is.nan(x)) return("::statcpp_test::kNaN")
    if (is.nan(x)) return("::statcpp_test::kNaN")
    if (is.infinite(x)) return(if (x > 0) "::statcpp_test::kInf" else "-::statcpp_test::kInf")
    sprintf("%a", x)
}

# Tolerances are readability-critical rather than precision-critical, so they are
# written as ordinary decimal literals.
cxx_tol <- function(x) {
    s <- sprintf("%g", x)
    if (!grepl("[.e]", s)) s <- paste0(s, ".0")
    s
}

# snake_case -> kPascalCase
cxx_name <- function(name) {
    parts <- strsplit(name, "_", fixed = TRUE)[[1]]
    parts <- parts[nzchar(parts)]
    paste0("k", paste0(toupper(substring(parts, 1, 1)), substring(parts, 2), collapse = ""))
}

# Render one named value as one or more `static constexpr` members.
render_member <- function(name, value) {
    k <- cxx_name(name)
    if (is.list(value)) {
        # Ragged groups: flatten plus a size array.
        flat <- unlist(value, use.names = FALSE)
        sizes <- vapply(value, length, integer(1))
        return(c(
            sprintf("    static constexpr double %sFlat[] = {%s};", k,
                    paste(vapply(flat, cxx_double, character(1)), collapse = ", ")),
            sprintf("    static constexpr std::size_t %sSizes[] = {%s};", k,
                    paste(sizes, collapse = ", ")),
            sprintf("    static constexpr std::size_t %sCount = %d;", k, length(value))
        ))
    }
    if (is.matrix(value)) {
        # Row-major flattening; statcpp takes vector<vector<double>> by row.
        flat <- as.vector(t(value))
        return(c(
            sprintf("    static constexpr double %sFlat[] = {%s};", k,
                    paste(vapply(flat, cxx_double, character(1)), collapse = ", ")),
            sprintf("    static constexpr std::size_t %sRows = %d;", k, nrow(value)),
            sprintf("    static constexpr std::size_t %sCols = %d;", k, ncol(value))
        ))
    }
    if (is.integer(value)) {
        if (length(value) == 1L) {
            return(sprintf("    static constexpr std::size_t %s = %d;", k, value))
        }
        return(sprintf("    static constexpr std::size_t %s[] = {%s};", k,
                       paste(value, collapse = ", ")))
    }
    if (length(value) == 1L) {
        return(sprintf("    static constexpr double %s = %s;", k, cxx_double(value)))
    }
    sprintf("    static constexpr double %s[] = {%s};", k,
            paste(vapply(value, cxx_double, character(1)), collapse = ", "))
}

# Register one reference case.
#
# @param struct  C++ struct name (PascalCase)
# @param note    How the reference was obtained, recorded in the header
# @param data    Inputs handed to statcpp
# @param expect  Reference results
# @param rtol    Relative tolerance
# @param atol    Absolute tolerance
# @param caveat  Optional note about a definition difference against R
# @param prtol Relative tolerance for quantities obtained through statcpp's own
#              distribution functions (p-values, quantile-derived bounds). These
#              are systematically looser than plain arithmetic because statcpp
#              and R use different incomplete beta and gamma implementations.
#              The 1e-8 default is calibrated from the largest deviation actually
#              observed across this suite (2.1e-09, the ANCOVA covariate p-value),
#              and still pins eight significant digits.
emit <- function(struct, note, data = list(), expect = list(),
                 rtol = 1e-12, atol = 0.0, prtol = 1e-8, caveat = NULL) {
    .env$cases[[length(.env$cases) + 1L]] <- list(
        struct = struct, note = note, data = data, expect = expect,
        rtol = rtol, atol = atol, prtol = prtol, caveat = caveat
    )
}

write_header <- function(path, guard_comment) {
    lines <- c(
        "/**",
        sprintf(" * @file %s", basename(path)),
        sprintf(" * @brief %s", guard_comment),
        " *",
        sprintf(" * Generated by generate_r_reference.R against R %s.", R_VERSION_USED),
        " * Do not edit by hand: re-run the script instead.",
        " *",
        " * Values are C++17 hexadecimal floating-point literals and are therefore",
        " * bit-identical to the values R produced.",
        " */",
        "",
        "#pragma once",
        "",
        "#include <cstddef>",
        "",
        "#include \"r_compare.hpp\"",
        "",
        "namespace statcpp_test::r_ref {",
        ""
    )
    for (c in .env$cases) {
        lines <- c(lines, "/**", sprintf(" * @brief %s", c$note))
        if (!is.null(c$caveat)) {
            lines <- c(lines, " *", strwrap(c$caveat, width = 96, prefix = " * "))
        }
        lines <- c(lines, " */", sprintf("struct %s {", c$struct))
        lines <- c(lines, sprintf("    static constexpr double kRtol = %s;", cxx_tol(c$rtol)))
        lines <- c(lines, sprintf("    static constexpr double kAtol = %s;", cxx_tol(c$atol)))
        lines <- c(lines, sprintf("    static constexpr double kPRtol = %s;", cxx_tol(c$prtol)))
        for (n in names(c$data))   lines <- c(lines, render_member(n, c$data[[n]]))
        for (n in names(c$expect)) lines <- c(lines, render_member(n, c$expect[[n]]))
        lines <- c(lines, "};", "")
    }
    lines <- c(lines, "}  // namespace statcpp_test::r_ref")
    writeLines(lines, path)
    cat(sprintf("wrote %s (%d cases)\n", path, length(.env$cases)))
    .env$cases <- list()
}

# Register a case that sweeps one function over a grid.
#
# Points where R returns zero or a non-finite value are dropped: a reference of
# exactly zero cannot be matched under a relative tolerance, and an underflowed
# tail carries no information about agreement.
#
# @param params Distribution parameters, emitted alongside the grid
# @param x      Grid of evaluation points
# @param expect Values R produced at those points
emit_grid <- function(struct, note, params = list(), x, expect,
                      rtol = 1e-12, prtol = 1e-8, caveat = NULL, xname = "x") {
    keep <- is.finite(expect) & expect != 0
    if (!any(keep)) stop(sprintf("emit_grid(%s): every grid point was dropped", struct))
    grid <- x[keep]
    if (is.integer(x)) grid <- as.integer(grid)
    data <- c(stats::setNames(list(grid), xname), params)
    emit(struct, note, data = data, expect = list(y = expect[keep]),
         rtol = rtol, prtol = prtol, caveat = caveat)
}

# ---------------------------------------------------------------------------
# Shared data
# ---------------------------------------------------------------------------

x1   <- c(5.1, 4.9, 5.3, 5.1, 4.8, 5.2, 5.0, 4.9)
g1   <- c(5.1, 4.9, 5.3, 5.1, 4.8)
g2   <- c(4.5, 4.7, 4.3, 4.6, 4.4)
pre  <- c(5.1, 4.9, 5.3, 5.1, 4.8, 5.2)
post <- c(5.5, 5.2, 5.6, 5.3, 5.1, 5.4)
pvec <- c(0.01, 0.04, 0.03, 0.20, 0.005)

# ---------------------------------------------------------------------------
# parametric_tests.hpp
# ---------------------------------------------------------------------------

r <- t.test(x1, mu = 5.0)
emit("TTestOneSample", "R: t.test(x, mu = 5.0)",
     data   = list(x = x1, mu0 = 5.0),
     expect = list(statistic = unname(r$statistic), df = unname(r$parameter),
                   p_value = r$p.value))

r <- t.test(x1, mu = 4.9, alternative = "greater")
emit("TTestOneSampleGreater", "R: t.test(x, mu = 4.9, alternative = \"greater\")",
     data   = list(x = x1, mu0 = 4.9),
     expect = list(statistic = unname(r$statistic), df = unname(r$parameter),
                   p_value = r$p.value))

r <- t.test(g1, g2, var.equal = TRUE)
emit("TTestTwoSample", "R: t.test(x, y, var.equal = TRUE)",
     data   = list(x = g1, y = g2),
     expect = list(statistic = unname(r$statistic), df = unname(r$parameter),
                   p_value = r$p.value))

r <- t.test(g1, g2)
emit("TTestWelch", "R: t.test(x, y)",
     data   = list(x = g1, y = g2),
     expect = list(statistic = unname(r$statistic), df = unname(r$parameter),
                   p_value = r$p.value))

r <- t.test(pre, post, paired = TRUE)
emit("TTestPaired", "R: t.test(x, y, paired = TRUE)",
     data   = list(x = pre, y = post),
     expect = list(statistic = unname(r$statistic), df = unname(r$parameter),
                   p_value = r$p.value))

# z_test has no base R counterpart; the reference is computed from the same
# closed form statcpp uses, with R supplying pnorm.
local({
    sigma <- 0.2
    mu0   <- 5.0
    z     <- (mean(x1) - mu0) / (sigma / sqrt(length(x1)))
    emit("ZTest", "R: manual, z = (mean(x) - mu0) / (sigma / sqrt(n)), p = 2 * pnorm(-|z|)",
         data   = list(x = x1, mu0 = mu0, sigma = sigma),
         expect = list(statistic = z, p_value = 2 * pnorm(-abs(z)), df = Inf))
})

r <- var.test(g1, g2)
emit("FTest", "R: var.test(x, y)",
     data   = list(x = g1, y = g2),
     expect = list(statistic = unname(r$statistic), df = unname(r$parameter[1]),
                   df2 = unname(r$parameter[2]), p_value = r$p.value))

local({
    obs <- c(16, 18, 16, 14, 12, 12)
    p   <- rep(1 / 6, 6)
    r   <- chisq.test(obs, p = p)
    emit("ChisqTestGof", "R: chisq.test(observed, p = rep(1/6, 6))",
         data   = list(observed = obs, expected = sum(obs) * p),
         expect = list(statistic = unname(r$statistic), df = unname(r$parameter),
                       p_value = r$p.value))
    r <- chisq.test(obs)
    emit("ChisqTestGofUniform", "R: chisq.test(observed)",
         data   = list(observed = obs),
         expect = list(statistic = unname(r$statistic), df = unname(r$parameter),
                       p_value = r$p.value))
})

local({
    m <- matrix(c(30, 20, 15, 25, 35, 20), nrow = 2, byrow = TRUE)
    r <- chisq.test(m, correct = FALSE)
    emit("ChisqTestIndependence", "R: chisq.test(table, correct = FALSE)",
         data   = list(table = m),
         expect = list(statistic = unname(r$statistic), df = unname(r$parameter),
                       p_value = r$p.value))
})

local({
    successes <- 45L; trials <- 100L; p0 <- 0.5
    z <- (successes / trials - p0) / sqrt(p0 * (1 - p0) / trials)
    emit("ZTestProportion",
         "R: manual, z = (p_hat - p0) / sqrt(p0 (1 - p0) / n); z^2 equals prop.test(correct = FALSE)",
         data   = list(successes = successes, trials = trials, p0 = p0),
         expect = list(statistic = z, p_value = 2 * pnorm(-abs(z)),
                       chisq_cross_check = unname(prop.test(successes, trials, p0,
                                                            correct = FALSE)$statistic)))
})

local({
    s1 <- 45L; n1 <- 100L; s2 <- 30L; n2 <- 90L
    pp <- (s1 + s2) / (n1 + n2)
    se <- sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z  <- (s1 / n1 - s2 / n2) / se
    emit("ZTestProportionTwoSample",
         "R: manual, pooled-proportion two-sample z; z^2 equals prop.test(correct = FALSE)",
         data   = list(successes1 = s1, trials1 = n1, successes2 = s2, trials2 = n2),
         expect = list(statistic = z, p_value = 2 * pnorm(-abs(z)),
                       chisq_cross_check = unname(prop.test(c(s1, s2), c(n1, n2),
                                                            correct = FALSE)$statistic)))
})

emit("BonferroniCorrection", "R: p.adjust(p, method = \"bonferroni\")",
     data = list(p_values = pvec), expect = list(adjusted = p.adjust(pvec, "bonferroni")))

emit("HolmCorrection", "R: p.adjust(p, method = \"holm\")",
     data = list(p_values = pvec), expect = list(adjusted = p.adjust(pvec, "holm")))

emit("BenjaminiHochbergCorrection", "R: p.adjust(p, method = \"BH\")",
     data = list(p_values = pvec), expect = list(adjusted = p.adjust(pvec, "BH")))

# ---------------------------------------------------------------------------
# nonparametric_tests.hpp
#
# Rank-based tests use values whose pairwise differences are exact in binary
# floating point, so that tie detection cannot diverge between R and C++ because
# of a one-ulp difference in a subtraction.
# ---------------------------------------------------------------------------

w1  <- c(12, 15, 11, 18, 14, 16, 13, 17)
mw1 <- c(12, 15, 11, 18, 14)
mw2 <- c(20, 17, 22, 19, 16)
kw  <- list(c(23, 25, 22, 27, 24), c(30, 28, 33, 31, 29), c(21, 19, 24, 22, 20))
# Levene needs groups whose spreads genuinely differ, otherwise the statistic is
# zero up to rounding and there is nothing meaningful to compare.
lev <- list(c(10.0, 10.5, 9.5, 10.2, 9.8), c(4.0, 16.0, 8.0, 12.0, 20.0),
            c(11.0, 9.0, 13.0, 7.0, 15.0))

r <- suppressWarnings(wilcox.test(w1, mu = 13.5, exact = FALSE, correct = TRUE))
emit("WilcoxonSignedRank", "R: wilcox.test(x, mu = 13.5, exact = FALSE, correct = TRUE)",
     data   = list(x = w1, mu0 = 13.5),
     expect = list(statistic = unname(r$statistic), p_value = r$p.value))

r <- suppressWarnings(wilcox.test(mw1, mw2, exact = FALSE, correct = TRUE))
emit("MannWhitneyU", "R: wilcox.test(x, y, exact = FALSE, correct = TRUE)",
     data   = list(x = mw1, y = mw2),
     expect = list(statistic = unname(r$statistic), p_value = r$p.value))

r <- kruskal.test(kw)
emit("KruskalWallis", "R: kruskal.test(groups)",
     data   = list(groups = kw),
     expect = list(statistic = unname(r$statistic), df = unname(r$parameter),
                   p_value = r$p.value))

# statcpp implements the Royston approximation; R uses the same family of
# approximations but not an identical polynomial, so the p-value needs a much
# looser bound than the rest of this file.
r <- shapiro.test(x1)
emit("ShapiroWilk", "R: shapiro.test(x)",
     data   = list(x = x1),
     expect = list(statistic = unname(r$statistic), p_value = r$p.value),
     rtol = 1e-6, prtol = 5e-2,
     caveat = "statcpp and R use different Shapiro-Wilk approximations; only agreement to a few percent is expected.")

# statcpp now uses the Dallal and Wilkinson (1986) analytic approximation, which
# is published for p <= 0.10. Three samples are used: two well inside that range,
# one just below the 0.05 boundary, and one above 0.10 where only D is compared.
lil_low  <- c(1, 1, 1, 1, 2, 2, 3, 5, 9, 20, 45)                                # p ~ 0.0024
lil_mid  <- c(0.05, 0.14, 4.59, 0.61, 0.35, 0.87, 2.55, 1.18, 0.67)             # p ~ 0.028
lil_high <- x1                                                                   # p ~ 0.73

r <- lillie.test(lil_low)
emit("LillieforsTest", "R: nortest::lillie.test(x), p ~ 0.002 (inside the published range)",
     data   = list(x = lil_low),
     expect = list(statistic = unname(r$statistic), p_value = r$p.value))

r <- lillie.test(lil_mid)
emit("LillieforsTestNearAlpha", "R: nortest::lillie.test(x), p ~ 0.028 (just below the 0.05 level)",
     data   = list(x = lil_mid),
     expect = list(statistic = unname(r$statistic), p_value = r$p.value))

r <- lillie.test(lil_high)
emit("LillieforsTestUpperRange",
     "R: nortest::lillie.test(x), p ~ 0.73; D statistic only",
     data   = list(x = lil_high),
     expect = list(statistic = unname(r$statistic)),
     caveat = paste(
         "The Dallal-Wilkinson approximation is published only for p <= 0.10, so above that",
         "level statcpp reports a value that indicates consistency with normality rather than",
         "an accurate p-value. Only the D statistic is compared here. Every conventional",
         "significance level lies inside the published range, which the two cases above cover."))

# statcpp's ks_test_normal forwards to lilliefors_test, so it shares the reference.
r <- lillie.test(lil_low)
emit("KsTestNormal",
     "R: nortest::lillie.test(x) (ks_test_normal forwards to lilliefors_test)",
     data   = list(x = lil_low),
     expect = list(statistic = unname(r$statistic), p_value = r$p.value))

r <- bartlett.test(lev)
emit("BartlettTest", "R: bartlett.test(groups)",
     data   = list(groups = lev),
     expect = list(statistic = unname(r$statistic), df = unname(r$parameter),
                   p_value = r$p.value))

local({
    y <- unlist(lev)
    g <- factor(rep(seq_along(lev), lengths(lev)))
    r <- leveneTest(y, g, center = median)
    emit("LeveneTest", "R: car::leveneTest(y, group, center = median)",
         data   = list(groups = lev),
         expect = list(statistic = r[["F value"]][1], p_value = r[["Pr(>F)"]][1]))
})

local({
    m <- matrix(c(8, 2, 1, 5), nrow = 2, byrow = TRUE)
    r <- fisher.test(m)
    emit("FisherExactTest", "R: fisher.test(matrix(c(a, b, c, d), nrow = 2, byrow = TRUE))",
         data   = list(a = 8L, b = 2L, c = 1L, d = 5L),
         expect = list(p_value = r$p.value))
})

local({
    v <- c(3, 1, 4, 1, 5, 9, 2, 6, 5, 3)
    emit("ComputeRanksWithTies", "R: rank(x, ties.method = \"average\")",
         data = list(x = v), expect = list(ranks = rank(v, ties.method = "average")))
})

# ---------------------------------------------------------------------------
# anova.hpp
# ---------------------------------------------------------------------------

anova_groups <- list(c(23, 25, 22, 27, 24), c(30, 28, 33, 31, 29), c(21, 19, 24, 22, 20))

local({
    y   <- unlist(anova_groups)
    g   <- factor(rep(seq_along(anova_groups), lengths(anova_groups)))
    fit <- aov(y ~ g)
    s   <- summary(fit)[[1]]
    ss_b <- s[["Sum Sq"]][1]; df_b <- s[["Df"]][1]; ms_b <- s[["Mean Sq"]][1]
    ss_w <- s[["Sum Sq"]][2]; df_w <- s[["Df"]][2]; ms_w <- s[["Mean Sq"]][2]
    ss_t <- ss_b + ss_w

    emit("OneWayAnova",
         "R: summary(aov(y ~ g)); eta^2, omega^2 and Cohen's f from the standard closed forms",
         data   = list(groups = anova_groups),
         expect = list(
             f_statistic = s[["F value"]][1], p_value = s[["Pr(>F)"]][1],
             ss_between = ss_b, df_between = df_b, ms_between = ms_b,
             ss_within = ss_w, df_within = df_w, ms_within = ms_w,
             ss_total = ss_t,
             eta_squared = ss_b / ss_t,
             omega_squared = (ss_b - df_b * ms_w) / (ss_t + ms_w),
             cohens_f = sqrt((ss_b / ss_t) / (1 - ss_b / ss_t))))

    means <- vapply(anova_groups, mean, numeric(1))
    sizes <- lengths(anova_groups)
    k     <- length(anova_groups)
    pairs <- which(upper.tri(matrix(0, k, k)), arr.ind = TRUE)
    pairs <- pairs[order(pairs[, "row"], pairs[, "col"]), , drop = FALSE]
    i_idx <- pairs[, "row"]; j_idx <- pairs[, "col"]
    se    <- sqrt(ms_w * (1 / sizes[i_idx] + 1 / sizes[j_idx]))
    diff  <- means[i_idx] - means[j_idx]
    ncomp <- nrow(pairs)

    # TukeyHSD reports "j - i"; statcpp reports mean[i] - mean[j] for i < j, so
    # the sign is flipped and the interval bounds are swapped.
    tk <- TukeyHSD(fit)$g
    emit("TukeyHsd", "R: TukeyHSD(aov(y ~ g))",
         data   = list(groups = anova_groups),
         expect = list(mean_diff = -tk[, "diff"], p_value = tk[, "p adj"],
                       lower = -tk[, "upr"], upper = -tk[, "lwr"]),
         rtol = 1e-6,
         caveat = "Sign flipped and bounds swapped relative to R: TukeyHSD reports mean[j] - mean[i]. The studentized range distribution is evaluated by different approximations, hence the looser tolerance.")

    t_stat <- diff / se
    emit("BonferroniPosthoc",
         "R: manual, pooled-MSE t statistic with Bonferroni adjustment (matches pairwise.t.test)",
         data   = list(groups = anova_groups),
         expect = list(mean_diff = diff, statistic = t_stat,
                       p_value = pmin(1, 2 * (1 - pt(abs(t_stat), df_w)) * ncomp),
                       cross_check = as.vector(pairwise.t.test(y, g,
                           p.adjust.method = "bonferroni")$p.value[!is.na(
                           pairwise.t.test(y, g, p.adjust.method = "bonferroni")$p.value)])))

    f_stat <- t_stat^2 / df_b
    emit("ScheffePosthoc", "R: manual, Scheffe F = t^2 / df_between, p = 1 - pf(F, df_between, df_error)",
         data   = list(groups = anova_groups),
         expect = list(mean_diff = diff, statistic = t_stat, f_statistic = f_stat,
                       p_value = 1 - pf(f_stat, df_b, df_w)),
         caveat = "statcpp stores the signed t statistic in posthoc_comparison::statistic and uses F = t^2 / df_between only to obtain the p-value.")

    ctrl   <- 1L
    others <- setdiff(seq_len(k), ctrl)
    se_c   <- sqrt(ms_w * (1 / sizes[others] + 1 / sizes[ctrl]))
    diff_c <- means[others] - means[ctrl]
    t_c    <- diff_c / se_c
    emit("DunnettPosthoc",
         "R: manual, Bonferroni-adjusted t against the control group",
         data   = list(groups = anova_groups, control_group = 0L),
         expect = list(mean_diff = diff_c, statistic = t_c,
                       p_value = pmin(1, 2 * (1 - pt(abs(t_c), df_w)) * length(others))),
         caveat = "statcpp's dunnett_posthoc applies a Bonferroni approximation, not the classical Dunnett procedure based on the multivariate t distribution. It therefore does not match multcomp::glht(..., mcp(g = \"Dunnett\")) and is compared against the closed form statcpp implements.")
})

local({
    # data[i][j] holds the observations for factor A level i and factor B level j.
    levels_a <- 2L; levels_b <- 3L; reps <- 3L
    # Deliberately non-additive: the A2 profile is not a constant shift of A1, so
    # the interaction sum of squares is genuinely non-zero.
    cells <- list(
        c(14, 16, 15), c(18, 20, 19), c(22, 21, 23),
        c(11, 13, 12), c(21, 20, 22), c(15, 16, 14))
    flat <- unlist(cells, use.names = FALSE)
    y <- flat
    a <- factor(rep(seq_len(levels_a), each = levels_b * reps))
    b <- factor(rep(rep(seq_len(levels_b), each = reps), times = levels_a))
    s <- summary(aov(y ~ a * b))[[1]]
    ss <- s[["Sum Sq"]]; df <- s[["Df"]]
    ss_e <- ss[4]
    emit("TwoWayAnova",
         "R: summary(aov(y ~ a * b)); partial eta^2 = SS_effect / (SS_effect + SS_error)",
         data   = list(values = flat, levels_a = levels_a, levels_b = levels_b, reps = reps),
         expect = list(
             f_a = s[["F value"]][1], p_a = s[["Pr(>F)"]][1],
             f_b = s[["F value"]][2], p_b = s[["Pr(>F)"]][2],
             f_interaction = s[["F value"]][3], p_interaction = s[["Pr(>F)"]][3],
             ss_a = ss[1], ss_b = ss[2], ss_interaction = ss[3], ss_error = ss_e,
             partial_eta_squared_a = ss[1] / (ss[1] + ss_e),
             partial_eta_squared_b = ss[2] / (ss[2] + ss_e),
             partial_eta_squared_interaction = ss[3] / (ss[3] + ss_e)))
})

local({
    # Each group is a list of (y, x) pairs; the covariate is entered first.
    y_g <- list(c(23, 25, 22, 27, 24), c(30, 28, 33, 31, 29), c(21, 19, 24, 22, 20))
    # The covariate must not determine y exactly, otherwise the error sum of
    # squares is zero and there is nothing meaningful to compare.
    x_g <- list(c(10, 12, 9, 15, 11), c(15, 13, 18, 16, 12), c(8, 6, 12, 9, 7))
    y <- unlist(y_g); x <- unlist(x_g)
    g <- factor(rep(seq_along(y_g), lengths(y_g)))
    # statcpp reports partial (type II) sums of squares for both the covariate
    # and the treatment, so the reference must not use R's sequential aov table.
    a <- Anova(lm(y ~ x + g), type = 2)
    emit("OneWayAncova", "R: car::Anova(lm(y ~ x + g), type = 2) (partial / type II sums of squares)",
         data   = list(y = y_g, x = x_g),
         expect = list(
             ss_covariate = a["x", "Sum Sq"], df_covariate = a["x", "Df"],
             f_covariate = a["x", "F value"], p_covariate = a["x", "Pr(>F)"],
             ss_treatment = a["g", "Sum Sq"], df_treatment = a["g", "Df"],
             f_treatment = a["g", "F value"], p_treatment = a["g", "Pr(>F)"],
             ss_error = a["Residuals", "Sum Sq"], df_error = a["Residuals", "Df"]),
         caveat = "statcpp uses partial (type II) sums of squares. Comparing against R's sequential summary(aov(y ~ x + g)) would mismatch on the covariate row.")
})

write_header("r_reference_hypothesis.hpp",
             "R reference values for hypothesis tests (parametric, nonparametric, ANOVA).")

# ===========================================================================
# Phase 2: distributions and special functions
#
# Each function is swept over a grid rather than checked at a single point, so
# that tail behaviour and the interior are both exercised.
# ===========================================================================

# ---------------------------------------------------------------------------
# continuous_distributions.hpp
# ---------------------------------------------------------------------------

xs <- function(a, b, by) seq(a, b, by = by)
ps <- seq(0.001, 0.999, by = 0.004)

local({
    mu <- 2.0; sigma <- 3.0; x <- xs(-16, 20, 0.25)
    emit_grid("NormalPdf", "R: dnorm(x, mean, sd)", list(mu = mu, sigma = sigma), x,
              dnorm(x, mu, sigma))
    emit_grid("NormalCdf", "R: pnorm(x, mean, sd)", list(mu = mu, sigma = sigma), x,
              pnorm(x, mu, sigma))
    emit_grid("NormalQuantile", "R: qnorm(p, mean, sd)", list(mu = mu, sigma = sigma), ps,
              qnorm(ps, mu, sigma), xname = "p", prtol = 1e-7,
              caveat = "Measured worst-case agreement with R over this grid is 7.1e-08 relative.")
})

local({
    df <- 7.0; x <- xs(-12, 12, 0.2)
    emit_grid("TPdf", "R: dt(x, df)", list(df = df), x, dt(x, df))
    emit_grid("TCdf", "R: pt(x, df)", list(df = df), x, pt(x, df))
    emit_grid("TQuantile", "R: qt(p, df)", list(df = df), ps, qt(ps, df), xname = "p")
})

local({
    df <- 5.0; x <- xs(0.05, 40, 0.25)
    emit_grid("ChisqPdf", "R: dchisq(x, df)", list(df = df), x, dchisq(x, df))
    emit_grid("ChisqCdf", "R: pchisq(x, df)", list(df = df), x, pchisq(x, df))
    emit_grid("ChisqQuantile", "R: qchisq(p, df)", list(df = df), ps, qchisq(ps, df), xname = "p")
})

local({
    df1 <- 4.0; df2 <- 9.0; x <- xs(0.02, 20, 0.1)
    emit_grid("FPdf", "R: df(x, df1, df2)", list(df1 = df1, df2 = df2), x, df(x, df1, df2))
    emit_grid("FCdf", "R: pf(x, df1, df2)", list(df1 = df1, df2 = df2), x, pf(x, df1, df2))
    emit_grid("FQuantile", "R: qf(p, df1, df2)", list(df1 = df1, df2 = df2), ps,
              qf(ps, df1, df2), xname = "p")
})

local({
    shape <- 2.5; rate <- 1.3; x <- xs(0.02, 25, 0.1)
    emit_grid("GammaPdf", "R: dgamma(x, shape, rate)", list(shape = shape, rate = rate), x,
              dgamma(x, shape, rate))
    emit_grid("GammaCdf", "R: pgamma(x, shape, rate)", list(shape = shape, rate = rate), x,
              pgamma(x, shape, rate))
    emit_grid("GammaQuantile", "R: qgamma(p, shape, rate)", list(shape = shape, rate = rate), ps,
              qgamma(ps, shape, rate), xname = "p")
})

local({
    a <- 2.0; b <- 5.0; x <- xs(0.002, 0.998, 0.004)
    emit_grid("BetaPdf", "R: dbeta(x, shape1, shape2)", list(alpha = a, beta_param = b), x,
              dbeta(x, a, b))
    emit_grid("BetaCdf", "R: pbeta(x, shape1, shape2)", list(alpha = a, beta_param = b), x,
              pbeta(x, a, b))
    emit_grid("BetaQuantile", "R: qbeta(p, shape1, shape2)", list(alpha = a, beta_param = b), ps,
              qbeta(ps, a, b), xname = "p")
})

local({
    lambda <- 1.7; x <- xs(0.01, 12, 0.05)
    emit_grid("ExponentialPdf", "R: dexp(x, rate)", list(lambda = lambda), x, dexp(x, lambda))
    emit_grid("ExponentialCdf", "R: pexp(x, rate)", list(lambda = lambda), x, pexp(x, lambda))
    emit_grid("ExponentialQuantile", "R: qexp(p, rate)", list(lambda = lambda), ps,
              qexp(ps, lambda), xname = "p")
})

local({
    a <- -1.0; b <- 3.0; x <- xs(-2, 4, 0.05)
    emit_grid("UniformPdf", "R: dunif(x, min, max)", list(a = a, b = b), x, dunif(x, a, b))
    emit_grid("UniformCdf", "R: punif(x, min, max)", list(a = a, b = b), x, punif(x, a, b))
    emit_grid("UniformQuantile", "R: qunif(p, min, max)", list(a = a, b = b), ps,
              qunif(ps, a, b), xname = "p")
})

local({
    mu <- 0.4; sigma <- 0.9; x <- xs(0.02, 30, 0.1)
    emit_grid("LognormalPdf", "R: dlnorm(x, meanlog, sdlog)", list(mu = mu, sigma = sigma), x,
              dlnorm(x, mu, sigma))
    emit_grid("LognormalCdf", "R: plnorm(x, meanlog, sdlog)", list(mu = mu, sigma = sigma), x,
              plnorm(x, mu, sigma))
    emit_grid("LognormalQuantile", "R: qlnorm(p, meanlog, sdlog)", list(mu = mu, sigma = sigma),
              ps, qlnorm(ps, mu, sigma), xname = "p")
})

local({
    shape <- 1.8; scale <- 2.2; x <- xs(0.02, 14, 0.05)
    emit_grid("WeibullPdf", "R: dweibull(x, shape, scale)", list(shape = shape, scale = scale), x,
              dweibull(x, shape, scale))
    emit_grid("WeibullCdf", "R: pweibull(x, shape, scale)", list(shape = shape, scale = scale), x,
              pweibull(x, shape, scale))
    emit_grid("WeibullQuantile", "R: qweibull(p, shape, scale)",
              list(shape = shape, scale = scale), ps, qweibull(ps, shape, scale), xname = "p")
})

local({
    k <- 3.0; df <- 12.0; q <- xs(0.1, 9, 0.1); pq <- seq(0.05, 0.995, by = 0.005)
    emit_grid("StudentizedRangeCdf", "R: ptukey(q, nmeans, df)", list(k = k, df = df), q,
              ptukey(q, k, df), xname = "q")
    emit_grid("StudentizedRangeQuantile", "R: qtukey(p, nmeans, df)", list(k = k, df = df), pq,
              qtukey(pq, k, df), xname = "p", prtol = 1e-6,
              caveat = "statcpp inverts its own studentized range CDF by root finding; measured worst-case agreement is 4.3e-07 relative.")
})

# ---------------------------------------------------------------------------
# discrete_distributions.hpp
# ---------------------------------------------------------------------------

pd <- seq(0.01, 0.99, by = 0.01)

local({
    n <- 20L; p <- 0.35; k <- 0:25
    emit_grid("BinomialPmf", "R: dbinom(k, size, prob)", list(n = n, p = p), k,
              dbinom(k, n, p), xname = "k")
    emit_grid("BinomialCdf", "R: pbinom(k, size, prob)", list(n = n, p = p), k,
              pbinom(k, n, p), xname = "k")
    emit_grid("BinomialQuantile", "R: qbinom(prob, size, prob)", list(n = n, p = p), pd,
              as.double(qbinom(pd, n, p)), xname = "prob")
})

local({
    lambda <- 4.2; k <- 0:30
    emit_grid("PoissonPmf", "R: dpois(k, lambda)", list(lambda = lambda), k, dpois(k, lambda),
              xname = "k")
    emit_grid("PoissonCdf", "R: ppois(k, lambda)", list(lambda = lambda), k, ppois(k, lambda),
              xname = "k")
    emit_grid("PoissonQuantile", "R: qpois(p, lambda)", list(lambda = lambda), pd,
              as.double(qpois(pd, lambda)), xname = "p")
})

local({
    p <- 0.3; k <- 0:30
    emit_grid("GeometricPmf", "R: dgeom(k, prob), k counts failures", list(p = p), k, dgeom(k, p),
              xname = "k")
    emit_grid("GeometricCdf", "R: pgeom(k, prob)", list(p = p), k, pgeom(k, p), xname = "k")
    emit_grid("GeometricQuantile", "R: qgeom(prob, prob)", list(p = p), pd,
              as.double(qgeom(pd, p)), xname = "prob")
})

local({
    N <- 50L; K <- 20L; n <- 12L; k <- 0:12
    emit_grid("HypergeomPmf", "R: dhyper(k, m = K, n = N - K, k = n)", list(cap_n = N, cap_k = K, n = n),
              k, dhyper(k, K, N - K, n), xname = "k")
    emit_grid("HypergeomCdf", "R: phyper(k, m = K, n = N - K, k = n)", list(cap_n = N, cap_k = K, n = n),
              k, phyper(k, K, N - K, n), xname = "k")
    emit_grid("HypergeomQuantile", "R: qhyper(p, m = K, n = N - K, k = n)",
              list(cap_n = N, cap_k = K, n = n), pd, as.double(qhyper(pd, K, N - K, n)), xname = "p")
})

local({
    r <- 5.0; p <- 0.4; k <- 0:40
    emit_grid("NbinomPmf", "R: dnbinom(k, size = r, prob = p), k counts failures",
              list(r = r, p = p), k, dnbinom(k, r, p), xname = "k")
    emit_grid("NbinomCdf", "R: pnbinom(k, size = r, prob = p)", list(r = r, p = p), k,
              pnbinom(k, r, p), xname = "k")
    emit_grid("NbinomQuantile", "R: qnbinom(prob, size = r, prob = p)", list(r = r, p = p), pd,
              as.double(qnbinom(pd, r, p)), xname = "prob")
})

local({
    p <- 0.4; k <- 0:1
    emit_grid("BernoulliPmf", "R: dbinom(k, size = 1, prob)", list(p = p), k, dbinom(k, 1, p),
              xname = "k")
    emit_grid("BernoulliCdf", "R: pbinom(k, size = 1, prob)", list(p = p), k, pbinom(k, 1, p),
              xname = "k")
    emit_grid("BernoulliQuantile", "R: qbinom(prob, size = 1, prob)", list(p = p), pd,
              as.double(qbinom(pd, 1, p)), xname = "prob")
})

local({
    # R has no discrete uniform; the closed forms are used directly.
    a <- 2L; b <- 9L; k <- 0:11; m <- b - a + 1
    emit_grid("DiscreteUniformPmf", "R: manual, 1 / (b - a + 1) on the support",
              list(a = a, b = b), k, ifelse(k >= a & k <= b, 1 / m, 0), xname = "k")
    emit_grid("DiscreteUniformCdf", "R: manual, (k - a + 1) / (b - a + 1) clamped to [0, 1]",
              list(a = a, b = b), k, pmin(1, pmax(0, (k - a + 1) / m)), xname = "k")
    emit_grid("DiscreteUniformQuantile", "R: manual, a + ceiling(p (b - a + 1)) - 1",
              list(a = a, b = b), pd, as.double(a + ceiling(pd * m) - 1), xname = "p")
})

local({
    n <- as.integer(c(0, 1, 2, 5, 5, 10, 10, 20, 30, 40, 52, 60, 100, 150, 170))
    k <- as.integer(c(0, 1, 1, 0, 2,  3,  5,  7, 15, 20, 5,  30,  50,  75,  85))
    emit("BinomialCoef", "R: choose(n, k)", data = list(n = n, k = k),
         expect = list(y = choose(n, k)))
    emit("LogBinomialCoef", "R: lchoose(n, k)", data = list(n = n, k = k),
         expect = list(y = lchoose(n, k)), rtol = 1e-11,
         caveat = "lchoose is evaluated through lgamma, so it is one tier looser than exact arithmetic.")
    nf <- as.integer(c(1, 2, 3, 5, 10, 20, 50, 100, 170, 250, 500, 1000))
    emit("LogFactorial", "R: lfactorial(n)", data = list(n = nf),
         expect = list(y = lfactorial(nf)), rtol = 1e-11)
})

# ---------------------------------------------------------------------------
# special_functions.hpp
# ---------------------------------------------------------------------------

local({
    x <- xs(-6, 6, 0.05)
    emit_grid("Erf", "R: 2 * pnorm(x * sqrt(2)) - 1", list(), x, 2 * pnorm(x * sqrt(2)) - 1)
    xe <- xs(-6, 26, 0.05)
    emit_grid("Erfc", "R: 2 * pnorm(-x * sqrt(2))", list(), xe, 2 * pnorm(-xe * sqrt(2)))

    xg <- c(xs(0.02, 20, 0.02), xs(20.5, 170, 0.5))
    emit_grid("Lgamma", "R: lgamma(x)", list(), xg, lgamma(xg))
    xt <- xs(0.02, 170, 0.02)
    emit_grid("Tgamma", "R: gamma(x)", list(), xt, gamma(xt))

    # The full left tail is exercised: statcpp evaluates the CDF through erfc, so
    # it stays accurate down to the smallest representable results.
    xn <- xs(-38, 8, 0.05)
    emit_grid("NormCdf", "R: pnorm(x), swept into the far left tail", list(), xn, pnorm(xn),
              prtol = 1e-11,
              caveat = "Measured worst-case agreement with R over this grid is 1.9e-13 relative, reached in the extreme tail.")
    emit_grid("NormQuantile", "R: qnorm(p)", list(), ps, qnorm(ps), xname = "p")

    # The survival function is swept over the whole upper tail: this is exactly
    # where forming the complement as 1 - norm_cdf(x) used to collapse to zero.
    xsf <- xs(-8, 38, 0.05)
    emit_grid("NormSf", "R: pnorm(x, lower.tail = FALSE)", list(), xsf,
              pnorm(xsf, lower.tail = FALSE), prtol = 1e-11,
              caveat = "Sweeps into the far upper tail, where R still returns representable values down to about 1e-299.")
})

local({
    a <- c(0.5, 1, 1.5, 2, 3, 5, 7.5, 10, 20, 50, 0.25, 4.5)
    b <- c(0.5, 2, 3.5, 1, 4, 5, 2.5, 30,  3, 10, 8.0, 0.75)
    emit("BetaFunction", "R: beta(a, b)", data = list(a = a, b = b),
         expect = list(y = beta(a, b)), rtol = 1e-11)
    emit("Lbeta", "R: lbeta(a, b)", data = list(a = a, b = b),
         expect = list(y = lbeta(a, b)), rtol = 1e-11)
})

local({
    a <- 2.5; b <- 4.0; x <- xs(0.002, 0.998, 0.004)
    emit_grid("Betainc", "R: pbeta(x, a, b), the regularized incomplete beta I_x(a, b)",
              list(a = a, b = b), x, pbeta(x, a, b))
    emit_grid("Betaincinv", "R: qbeta(p, a, b)", list(a = a, b = b), ps, qbeta(ps, a, b),
              xname = "p")
})

local({
    a <- 3.5; x <- xs(0.02, 30, 0.05)
    emit_grid("GammaincLower", "R: pgamma(x, a), the regularized P(a, x)", list(a = a), x,
              pgamma(x, a))
    emit_grid("GammaincUpper", "R: pgamma(x, a, lower.tail = FALSE), the regularized Q(a, x)",
              list(a = a), x, pgamma(x, a, lower.tail = FALSE), prtol = 1e-6,
              caveat = "The far upper tail is computed by a continued fraction; measured worst-case agreement is 3.1e-07 relative.")
    emit_grid("GammaincLowerInv", "R: qgamma(p, a)", list(a = a), ps, qgamma(ps, a), xname = "p")
})

write_header("r_reference_distributions.hpp",
             "R reference values for continuous, discrete and special functions.")

# ===========================================================================
# Phase 3: descriptive statistics
#
# This group carries the most definition differences against R, so each case
# records which R form it is compared with.
# ===========================================================================

dv  <- c(4.2, 1.7, 8.9, 3.3, 6.1, 2.8, 7.4, 5.5, 9.6, 3.9, 6.7, 2.1)
dvs <- sort(dv)
wv  <- c(2, 5, 1, 3, 4, 2, 6, 1, 3, 2, 4, 5)
nv  <- length(dv)

# ---------------------------------------------------------------------------
# basic_statistics.hpp
# ---------------------------------------------------------------------------

emit("BasicStatistics",
     "R: sum, mean, median, exp(mean(log)), 1/mean(1/x), mean(trim = 0.2), which.max/which.min",
     data   = list(x = dv, x_sorted = dvs, trim = 0.2),
     expect = list(
         sum = sum(dv), mean = mean(dv), median = median(dvs),
         geometric_mean = exp(mean(log(dv))), harmonic_mean = 1 / mean(1 / dv),
         trimmed_mean = mean(dvs, trim = 0.2),
         argmax = as.double(which.max(dv) - 1L), argmin = as.double(which.min(dv) - 1L)),
     caveat = "median, percentile and the order statistics require a sorted range in statcpp, so x_sorted is supplied. argmax/argmin are zero-based in C++ and one-based in R.")

local({
    uniq <- c(3, 7, 3, 5, 9, 3, 7, 1, 5, 7, 3, 2)   # 3 occurs four times
    ties <- c(1, 1, 2, 2, 3)                        # 1 and 2 both occur twice
    tb <- table(uniq)
    emit("Mode", "R: the most frequent value of table(x)",
         data = list(x = uniq),
         expect = list(mode = as.double(names(tb)[which.max(tb)])),
         caveat = "statcpp iterates an ordered map, so a tie resolves to the smallest value. See Modes for the tied case.")
    tb2 <- table(ties)
    emit("Modes", "R: every value attaining the maximum of table(x)",
         data = list(x = ties),
         expect = list(modes = as.double(names(tb2)[tb2 == max(tb2)])))
})

emit("WeightedBasicStatistics", "R: weighted.mean(x, w) and sum(w) / sum(w / x)",
     data   = list(x = dv, w = wv),
     expect = list(weighted_mean = weighted.mean(dv, wv),
                   weighted_harmonic_mean = sum(wv) / sum(wv / dv)))

local({
    a <- 3.5; b <- 8.2
    emit("LogarithmicMean", "R: (b - a) / (log(b) - log(a))",
         data = list(a = a, b = b), expect = list(y = (b - a) / (log(b) - log(a))))
})

# ---------------------------------------------------------------------------
# dispersion_spread.hpp
# ---------------------------------------------------------------------------

emit("Dispersion",
     "R: var, sd, IQR(type = 7), diff(range), mean(abs(x - mean(x))), sd/|mean|, exp(sd(log(x)))",
     data   = list(x = dv, x_sorted = dvs),
     expect = list(
         var_ddof0 = var(dv) * (nv - 1) / nv,
         var_ddof1 = var(dv),
         variance = var(dv),
         sample_variance = var(dv),
         population_variance = var(dv) * (nv - 1) / nv,
         stdev_ddof0 = sd(dv) * sqrt((nv - 1) / nv),
         stdev_ddof1 = sd(dv),
         stddev = sd(dv),
         sample_stddev = sd(dv),
         population_stddev = sd(dv) * sqrt((nv - 1) / nv),
         range = diff(range(dv)),
         iqr = IQR(dvs, type = 7),
         mean_absolute_deviation = mean(abs(dv - mean(dv))),
         coefficient_of_variation = sd(dv) / abs(mean(dv)),
         geometric_stddev = exp(sd(log(dv)))),
     caveat = "statcpp's var/stdev take a ddof argument defaulting to 0, so the bare call is the population form, unlike R's var/sd. mean_absolute_deviation is the mean absolute deviation about the mean and is NOT R's mad(), which is the median absolute deviation.")

local({
    cw <- cov.wt(cbind(dv), wt = wv, method = "unbiased")
    emit("WeightedDispersion", "R: cov.wt(x, wt, method = \"unbiased\")",
         data   = list(x = dv, w = wv),
         expect = list(weighted_variance = cw$cov[1, 1],
                       weighted_stddev = sqrt(cw$cov[1, 1])),
         caveat = "statcpp divides by (sum(w) - sum(w^2)/sum(w)), which is algebraically identical to the reliability-weight estimator base R uses in cov.wt(method = \"unbiased\"). It differs from Hmisc::wtd.var, which treats the weights as frequencies.")
})

# ---------------------------------------------------------------------------
# order_statistics.hpp
# ---------------------------------------------------------------------------

local({
    pp <- seq(0, 1, by = 0.01)
    emit_grid("Percentile", "R: quantile(x, p, type = 7)", list(x = dvs), pp,
              as.double(quantile(dvs, pp, type = 7)), xname = "p")
    q <- as.double(quantile(dvs, c(0.25, 0.5, 0.75), type = 7))
    emit("OrderStatistics",
         "R: min, max, quantile(type = 7) for the quartiles and the five-number summary",
         data   = list(x = dvs),
         expect = list(minimum = min(dvs), maximum = max(dvs),
                       q1 = q[1], q2 = q[2], q3 = q[3],
                       fns_min = min(dvs), fns_q1 = q[1], fns_median = q[2],
                       fns_q3 = q[3], fns_max = max(dvs)),
         caveat = "five_number_summary uses type 7 quantiles, NOT R's fivenum(), which reports Tukey hinges and would disagree.")
})

local({
    # statcpp's weighted percentile is a step function with averaging when the
    # cumulative weight lands exactly on the target. The closed form is
    # reproduced here so the comparison tests the implementation, not a guess.
    wq <- function(x, w, p) {
        o <- order(x); x <- x[o]; w <- w[o]
        tw <- sum(w)
        if (p <= 0) return(x[1])
        if (p >= 1) return(x[length(x)])
        target <- p * tw
        tol <- .Machine$double.eps * tw
        cum <- 0
        for (i in seq_along(x)) {
            cum <- cum + w[i]
            if (cum >= target) {
                if (abs(cum - target) <= tol && i < length(x)) return((x[i] + x[i + 1]) / 2)
                return(x[i])
            }
        }
        x[length(x)]
    }
    pw <- c(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
    emit("WeightedOrderStatistics",
         "R: manual, statcpp's step-function weighted quantile with averaging at exact hits",
         data   = list(x = dv, w = wv, p = pw),
         expect = list(weighted_median = wq(dv, wv, 0.5),
                       weighted_percentile = vapply(pw, function(p) wq(dv, wv, p), numeric(1)),
                       unit_weight_check = vapply(pw, function(p) wq(dv, rep(1, nv), p), numeric(1)),
                       quantile_type2 = as.double(quantile(dv, pw, type = 2))),
         caveat = "With unit weights the definition coincides with quantile(type = 2); unit_weight_check and quantile_type2 record that equivalence.")
})

# ---------------------------------------------------------------------------
# shape_of_distribution.hpp
# ---------------------------------------------------------------------------

emit("Shape", "R: e1071::skewness / e1071::kurtosis, type 1 for population and type 2 for sample",
     data   = list(x = dv),
     expect = list(
         population_skewness = e1071::skewness(dv, type = 1),
         sample_skewness = e1071::skewness(dv, type = 2),
         skewness = e1071::skewness(dv, type = 2),
         population_kurtosis = e1071::kurtosis(dv, type = 1),
         sample_kurtosis = e1071::kurtosis(dv, type = 2),
         kurtosis = e1071::kurtosis(dv, type = 2)),
     caveat = "Both kurtosis forms are excess kurtosis, so a normal sample gives about zero. skewness and kurtosis are aliases for the sample forms.")

# ---------------------------------------------------------------------------
# correlation_covariance.hpp
# ---------------------------------------------------------------------------

local({
    xa <- c(12.1, 15.3, 9.8, 18.2, 14.0, 16.7, 11.4, 19.5, 13.2, 17.1)
    ya <- c(23.4, 28.9, 19.1, 34.2, 26.5, 31.0, 21.8, 36.7, 25.1, 32.3)
    n  <- length(xa)
    emit("Correlation",
         "R: cor(pearson/spearman/kendall), cov, and the population form cov * (n - 1) / n",
         data   = list(x = xa, y = ya),
         expect = list(
             pearson_correlation = cor(xa, ya, method = "pearson"),
             spearman_correlation = cor(xa, ya, method = "spearman"),
             kendall_tau = cor(xa, ya, method = "kendall"),
             covariance = cov(xa, ya),
             sample_covariance = cov(xa, ya),
             population_covariance = cov(xa, ya) * (n - 1) / n),
         caveat = "kendall_tau counts ties in x, y and both, so it is tau-b, matching R's default.")

    ww <- c(3, 1, 4, 2, 5, 2, 3, 1, 4, 2)
    cw <- cov.wt(cbind(xa, ya), wt = ww, method = "unbiased")
    emit("WeightedCovariance", "R: cov.wt(cbind(x, y), wt, method = \"unbiased\")",
         data   = list(x = xa, y = ya, w = ww),
         expect = list(weighted_covariance = cw$cov[1, 2]))
})

# ---------------------------------------------------------------------------
# distance_metrics.hpp
# ---------------------------------------------------------------------------

local({
    va <- c(1.5, 3.2, 0.8, 4.7, 2.1, 5.9)
    vb <- c(2.8, 1.1, 3.4, 2.2, 4.6, 3.3)
    m  <- rbind(va, vb)
    cs <- sum(va * vb) / (sqrt(sum(va^2)) * sqrt(sum(vb^2)))
    emit("Distance",
         "R: dist(euclidean/manhattan/maximum/minkowski) and the cosine similarity closed form",
         data   = list(x = va, y = vb, p = 3.0),
         expect = list(
             euclidean_distance = as.double(dist(m, method = "euclidean")),
             manhattan_distance = as.double(dist(m, method = "manhattan")),
             chebyshev_distance = as.double(dist(m, method = "maximum")),
             minkowski_distance = as.double(dist(m, method = "minkowski", p = 3)),
             cosine_similarity = cs,
             cosine_distance = 1 - cs,
             proxy_cosine_check = as.double(proxy::simil(m, method = "cosine"))),
         caveat = "Chebyshev is R's \"maximum\" metric. proxy_cosine_check confirms the cosine closed form against proxy::simil.")
})

local({
    # statcpp's mahalanobis_distance documents that only two dimensions are
    # supported and throws otherwise, so the reference is two-dimensional.
    x  <- c(2.5, 3.1)
    mu <- c(2.0, 3.0)
    S  <- matrix(c(1.2, 0.3,
                   0.3, 0.9), nrow = 2, byrow = TRUE)
    emit("Mahalanobis", "R: sqrt(mahalanobis(x, center, cov)) on 2-dimensional data",
         data   = list(x = x, mean = mu, cov = S),
         expect = list(y = sqrt(mahalanobis(x, mu, S))),
         caveat = "R's mahalanobis() returns the squared distance; statcpp returns the distance itself, so the reference takes a square root. statcpp supports two dimensions only.")
})

# ---------------------------------------------------------------------------
# frequency_distribution.hpp
# ---------------------------------------------------------------------------

local({
    fv <- c(3, 7, 3, 5, 9, 3, 7, 1, 5, 7, 3, 2, 9, 5, 3)
    tb <- table(fv)
    values <- as.double(names(tb))
    counts <- as.double(tb)
    emit("Frequency",
         "R: table, prop.table(table), cumsum(table), cumsum(prop.table(table))",
         data   = list(x = fv),
         expect = list(
             values = values,
             count = counts,
             relative = as.double(prop.table(tb)),
             cumulative_count = as.double(cumsum(tb)),
             cumulative_relative = as.double(cumsum(prop.table(tb))),
             total = as.double(length(fv))),
         caveat = "statcpp returns these keyed by value in ascending order, matching table()'s ordering.")
})

write_header("r_reference_descriptive.hpp",
             "R reference values for descriptive statistics, correlation, distance and frequency.")

# ===========================================================================
# Phase 4a: regression, GLM and model selection
# ===========================================================================

# The response must carry genuine scatter: an almost exactly linear sample drives
# the residuals and the p-values to zero, where a relative comparison is vacuous.
reg_x <- c(1.2, 2.5, 3.1, 4.8, 5.3, 6.7, 7.2, 8.9, 9.4, 10.1, 11.6, 12.3)
reg_y <- c(3.8, 4.9, 7.2, 8.6, 12.1, 12.4, 16.0, 17.2, 18.4, 21.9, 22.1, 26.3)

local({
    fit <- lm(reg_y ~ reg_x)
    s   <- summary(fit)
    n   <- length(reg_x)
    ss_res <- sum(residuals(fit)^2)
    ss_tot <- sum((reg_y - mean(reg_y))^2)
    emit("SimpleLinearRegression", "R: lm(y ~ x) and summary(lm)",
         data   = list(x = reg_x, y = reg_y),
         expect = list(
             intercept = unname(coef(fit)[1]), slope = unname(coef(fit)[2]),
             intercept_se = unname(s$coefficients[1, 2]), slope_se = unname(s$coefficients[2, 2]),
             intercept_t = unname(s$coefficients[1, 3]), slope_t = unname(s$coefficients[2, 3]),
             intercept_p = unname(s$coefficients[1, 4]), slope_p = unname(s$coefficients[2, 4]),
             r_squared = s$r.squared, adj_r_squared = s$adj.r.squared,
             residual_se = s$sigma,
             f_statistic = unname(s$fstatistic[1]),
             f_p_value = unname(pf(s$fstatistic[1], s$fstatistic[2], s$fstatistic[3],
                                   lower.tail = FALSE)),
             df_regression = unname(s$fstatistic[2]), df_residual = unname(s$fstatistic[3]),
             ss_total = ss_tot, ss_regression = ss_tot - ss_res, ss_residual = ss_res,
             predict_at = 7.5,
             predict_value = unname(predict(fit, data.frame(reg_x = 7.5))),
             r_squared_fn = s$r.squared,
             adjusted_r_squared_fn = s$adj.r.squared,
             press = sum((residuals(fit) / (1 - hatvalues(fit)))^2),
             aic_linear = AIC(fit), bic_linear = BIC(fit),
             n = as.integer(n)),
         prtol = 1e-7,
         caveat = "Measured worst-case agreement on the p-values is 1.9e-08 relative; they pass through statcpp's own incomplete beta function.")

    ci <- predict(fit, data.frame(reg_x = 7.5), interval = "confidence", level = 0.95)
    pi <- predict(fit, data.frame(reg_x = 7.5), interval = "prediction", level = 0.95)
    emit("RegressionIntervals",
         "R: predict(lm, interval = \"confidence\") and interval = \"prediction\"",
         data   = list(x = reg_x, y = reg_y, x_new = 7.5, confidence = 0.95),
         expect = list(ci_fit = unname(ci[1, 1]), ci_lower = unname(ci[1, 2]),
                       ci_upper = unname(ci[1, 3]),
                       pi_fit = unname(pi[1, 1]), pi_lower = unname(pi[1, 2]),
                       pi_upper = unname(pi[1, 3])))

    # statcpp's naming differs from R's: its "standardized" residual divides by
    # sigma alone, while its "studentized" residual is the internally studentized
    # form that R calls rstandard. statcpp has no externally studentized residual.
    emit("ResidualDiagnostics",
         "R: residuals, residuals/sigma, rstandard, hatvalues, cooks.distance and Durbin-Watson",
         data   = list(x = reg_x, y = reg_y),
         expect = list(
             residuals = unname(residuals(fit)),
             standardized = unname(residuals(fit)) / s$sigma,
             studentized = unname(rstandard(fit)),
             rstudent_r = unname(rstudent(fit)),
             hat_values = unname(hatvalues(fit)),
             cooks = unname(cooks.distance(fit)),
             durbin_watson = sum(diff(residuals(fit))^2) / sum(residuals(fit)^2)),
         caveat = "statcpp's standardized_residuals is e/sigma; its studentized_residuals is R's rstandard. rstudent_r records R's externally studentized residual, which statcpp does not provide.")
})

local({
    set.seed(0)
    x1 <- c(2.1, 3.4, 1.8, 5.2, 4.6, 6.9, 3.3, 7.1, 5.8, 8.4, 6.2, 9.1)
    x2 <- c(5.5, 4.1, 6.8, 3.2, 4.9, 2.6, 5.1, 2.2, 3.7, 1.9, 3.4, 1.4)
    y  <- c(13.1, 14.2, 11.8, 18.4, 18.0, 22.3, 14.9, 25.4, 19.6, 28.5, 22.2, 28.7)
    X  <- cbind(x1, x2)
    fit <- lm(y ~ x1 + x2)
    s   <- summary(fit)
    ss_res <- sum(residuals(fit)^2); ss_tot <- sum((y - mean(y))^2)
    emit("MultipleLinearRegression", "R: lm(y ~ x1 + x2) and summary(lm)",
         data   = list(x = X, y = y, x_new = c(5.0, 4.0)),
         expect = list(
             coefficients = unname(coef(fit)),
             coefficient_se = unname(s$coefficients[, 2]),
             t_statistics = unname(s$coefficients[, 3]),
             p_values = unname(s$coefficients[, 4]),
             r_squared = s$r.squared, adj_r_squared = s$adj.r.squared,
             residual_se = s$sigma,
             f_statistic = unname(s$fstatistic[1]),
             f_p_value = unname(pf(s$fstatistic[1], s$fstatistic[2], s$fstatistic[3],
                                   lower.tail = FALSE)),
             df_regression = unname(s$fstatistic[2]), df_residual = unname(s$fstatistic[3]),
             ss_total = ss_tot, ss_regression = ss_tot - ss_res, ss_residual = ss_res,
             predict_value = unname(predict(fit, data.frame(x1 = 5.0, x2 = 4.0))),
             aic_linear = AIC(fit), bic_linear = BIC(fit),
             n = as.integer(length(y)),
             vif = unname(car::vif(fit)),
             cor_determinant = det(cor(X)),
             loocv_mse = mean((residuals(fit) / (1 - hatvalues(fit)))^2)),
         caveat = "loocv_mse uses the algebraic shortcut for leave-one-out cross-validation, which is exact for linear models.")
})

local({
    # The classes must overlap: perfectly separable data makes the IRLS iteration
    # diverge and the maximum likelihood estimate does not exist.
    # Predictors are centred: statcpp forms X'WX explicitly and inverts it, which
    # squares the condition number, so an uncentred design of the same data loses
    # about five digits in the standard errors relative to R's QR-based covariance.
    x1 <- c(-7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5)
    x2 <- c(-1.6, -2.3, -0.2, -1.9, 0.5, -1.0, 1.3, -0.1, 2.1, 1.0, 3.2, 1.9, 0.3, 1.8, -0.5, 2.7)
    yb <- c(0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1)
    X  <- cbind(x1, x2)
    fit <- glm(yb ~ x1 + x2, family = binomial())
    s   <- summary(fit)
    nul <- glm(yb ~ 1, family = binomial())
    ci  <- exp(confint.default(fit, level = 0.95))
    xn  <- data.frame(x1 = 6.5, x2 = 4.2)
    emit("LogisticRegression", "R: glm(y ~ x1 + x2, family = binomial())",
         data   = list(x = X, y = yb, x_new = c(6.5, 4.2)),
         expect = list(
             coefficients = unname(coef(fit)),
             coefficient_se = unname(s$coefficients[, 2]),
             z_statistics = unname(s$coefficients[, 3]),
             p_values = unname(s$coefficients[, 4]),
             null_deviance = fit$null.deviance, residual_deviance = fit$deviance,
             df_null = fit$df.null, df_residual = fit$df.residual,
             aic = fit$aic,
             log_likelihood = as.numeric(logLik(fit)),
             null_log_likelihood = as.numeric(logLik(nul)),
             odds_ratios = unname(exp(coef(fit)))[-1],
             or_ci_lower = unname(ci[, 1])[-1], or_ci_upper = unname(ci[, 2])[-1],
             predict_probability = unname(predict(fit, xn, type = "response")),
             mcfadden = 1 - as.numeric(logLik(fit)) / as.numeric(logLik(nul)),
             nagelkerke = unname(performance::r2_nagelkerke(fit)),
             deviance_residuals = unname(residuals(fit, type = "deviance")),
             pearson_residuals = unname(residuals(fit, type = "pearson")),
             response_residuals = unname(residuals(fit, type = "response"))),
         rtol = 1e-7, prtol = 1e-4,
         caveat = "Measured agreement with R: deviance and AIC to 1.5e-16, coefficients to 1.5e-08, odds ratios to 2.4e-09, but standard errors only to 3.2e-05. statcpp's IRLS stops on the deviance, which is flat at the optimum, so the coefficients settle around sqrt(machine epsilon) and inverting X'WX amplifies that into the standard errors. odds_ratios and odds_ratios_ci exclude the intercept, so the reference drops the first coefficient. statcpp inverts X'WX explicitly, which is more sensitive to the conditioning of the design than R's QR-based covariance.")
})

local({
    x1 <- c(-5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
    x2 <- c(-1.7, -1.0, -1.4, -0.3, -1.1, 0.2, -0.6, 0.7, -0.1, 1.2, 0.4, 1.7)
    yc <- c(2, 3, 4, 6, 7, 11, 12, 18, 20, 27, 30, 41)
    X  <- cbind(x1, x2)
    fit <- glm(yc ~ x1 + x2, family = poisson())
    s   <- summary(fit)
    nul <- glm(yc ~ 1, family = poisson())
    xn  <- data.frame(x1 = 6.5, x2 = 1.7)
    pr  <- residuals(fit, type = "pearson")
    emit("PoissonRegression", "R: glm(y ~ x1 + x2, family = poisson())",
         data   = list(x = X, y = yc, x_new = c(6.5, 1.7)),
         expect = list(
             coefficients = unname(coef(fit)),
             coefficient_se = unname(s$coefficients[, 2]),
             z_statistics = unname(s$coefficients[, 3]),
             p_values = unname(s$coefficients[, 4]),
             null_deviance = fit$null.deviance, residual_deviance = fit$deviance,
             df_null = fit$df.null, df_residual = fit$df.residual,
             aic = fit$aic,
             log_likelihood = as.numeric(logLik(fit)),
             null_log_likelihood = as.numeric(logLik(nul)),
             incidence_rate_ratios = unname(exp(coef(fit)))[-1],
             predict_count = unname(predict(fit, xn, type = "response")),
             mcfadden = 1 - as.numeric(logLik(fit)) / as.numeric(logLik(nul)),
             overdispersion = sum(pr^2) / fit$df.residual),
         rtol = 1e-8, prtol = 1e-7,
         caveat = "The intercept p-value is about 1.3e-117. In the extreme tail a p-value is exponentially sensitive to its z statistic, so the 2.5e-09 difference in z becomes 5.9e-08 in p; that sets the tolerance here. Poisson agreement is otherwise far tighter than logistic: coefficients to 4.3e-15 and standard errors to 1.1e-10. incidence_rate_ratios excludes the intercept, so the reference drops the first coefficient. overdispersion is the Pearson chi-square divided by the residual degrees of freedom.")
})

local({
    ll <- -42.7; k <- 4L; n <- 30L
    emit("InformationCriteria",
         "R: closed forms, AIC = -2 logLik + 2k, BIC = -2 logLik + k log n, AICc adds 2k(k+1)/(n-k-1)",
         data   = list(log_likelihood = ll, k = k, n = n),
         expect = list(aic = -2 * ll + 2 * k,
                       bic = -2 * ll + k * log(n),
                       aicc = -2 * ll + 2 * k + 2 * k * (k + 1) / (n - k - 1)))
})

local({
    x1 <- c(2.1, 3.4, 1.8, 5.2, 4.6, 6.9, 3.3, 7.1, 5.8, 8.4, 6.2, 9.1)
    x2 <- c(5.5, 4.1, 6.8, 3.2, 4.9, 2.6, 5.1, 2.2, 3.7, 1.9, 3.4, 1.4)
    y  <- c(13.1, 14.2, 11.8, 18.4, 18.0, 22.3, 14.9, 25.4, 19.6, 28.5, 22.2, 28.7)
    X  <- cbind(x1, x2)
    lambda <- 1.5
    n <- nrow(X); p <- ncol(X)
    # statcpp standardises with the population standard deviation, centres y, then
    # solves the ridge normal equations by coordinate descent, which converges to
    # the closed form used here.
    xmean <- colMeans(X)
    xsd   <- apply(X, 2, function(v) sqrt(sum((v - mean(v))^2) / length(v)))
    Z  <- sweep(sweep(X, 2, xmean, "-"), 2, xsd, "/")
    yc <- y - mean(y)
    beta <- as.numeric(solve(t(Z) %*% Z + lambda * diag(p), t(Z) %*% yc))
    slopes <- beta / xsd
    intercept <- mean(y) - sum(slopes * xmean)
    fitted_vals <- intercept + as.numeric(X %*% slopes)
    emit("RidgeRegression",
         "R: closed-form ridge on population-standardised predictors with centred response",
         data   = list(x = X, y = y, lambda = lambda),
         expect = list(coefficients = c(intercept, slopes),
                       mse = mean((y - fitted_vals)^2)),
         prtol = 1e-5,
         caveat = "The coordinate descent stops at the API default tol = 1e-6, so agreement with the closed form is about 7.7e-07 relative. Not compared with MASS::lm.ridge, which scales the penalty differently. The reference reproduces the estimator statcpp defines: population standardisation, centred response, penalty lambda on the standardised scale.")

    emit("Multicollinearity", "R: car::vif, det(cor(X)) and 1 - |det(cor(X))|",
         data   = list(x = X),
         expect = list(vif = unname(car::vif(lm(y ~ x1 + x2))),
                       determinant = det(cor(X)),
                       score = 1 - abs(det(cor(X)))))
})

write_header("r_reference_regression.hpp",
             "R reference values for linear regression, GLM and model selection.")

# ===========================================================================
# Phase 4b: effect sizes, interval estimation and power
# ===========================================================================

es1 <- c(5.1, 4.9, 5.3, 5.1, 4.8, 5.2, 5.0, 4.9, 5.4, 5.2)
es2 <- c(4.5, 4.7, 4.3, 4.6, 4.4, 4.8, 4.2, 4.6, 4.4, 4.5)

local({
    n1 <- length(es1); n2 <- length(es2)
    sp <- sqrt(((n1 - 1) * var(es1) + (n2 - 1) * var(es2)) / (n1 + n2 - 2))
    d2 <- (mean(es1) - mean(es2)) / sp
    df2 <- n1 + n2 - 2
    j <- 1 - 3 / (4 * df2 - 1)
    mu0 <- 5.0
    d1 <- (mean(es1) - mu0) / sd(es1)
    j1 <- 1 - 3 / (4 * (n1 - 1) - 1)
    emit("EffectSizeSamples",
         "R: closed forms cross-checked against effectsize::cohens_d and effectsize::hedges_g",
         data   = list(x = es1, y = es2, mu0 = mu0),
         expect = list(
             cohens_d_one_sample = d1,
             hedges_g_one_sample = j1 * d1,
             cohens_d_two_sample = d2,
             hedges_g_two_sample = j * d2,
             glass_delta = (mean(es2) - mean(es1)) / sd(es1),
             hedges_correction_factor = j,
             effectsize_d_check = effectsize::cohens_d(es1, es2)$Cohens_d,
             effectsize_g_check = effectsize::hedges_g(es1, es2)$Hedges_g),
         caveat = "effectsize_g_check is NOT expected to match: the effectsize package uses the exact gamma-based correction factor, while statcpp uses the standard 1 - 3/(4 df - 1) approximation. The two differ by about 1e-04 here. glass_delta divides by the standard deviation of the first (control) sample and reports mean(treatment) - mean(control).")

    a <- 30; b <- 20; cc <- 15; d <- 35
    emit("EffectSizeConversions",
         "R: closed forms for the effect-size conversions and 2x2 measures",
         data   = list(d = 0.8, r = 0.5, t = 2.6, df = 18.0, f = 4.2, df1 = 2.0, df2 = 27.0,
                       p1 = 0.65, p2 = 0.40,
                       ss_effect = 120.5, ss_total = 480.0, ms_error = 9.5, df_effect = 3.0,
                       a = a, b = b, c = cc, d_cell = d),
         expect = list(
             d_to_r = 0.8 / sqrt(0.8^2 + 4),
             r_to_d = 2 * 0.5 / sqrt(1 - 0.5^2),
             t_to_r = 2.6 / sqrt(2.6^2 + 18),
             partial_eta_squared = (4.2 * 2) / (4.2 * 2 + 27),
             cohens_h = 2 * (asin(sqrt(0.65)) - asin(sqrt(0.40))),
             eta_squared = 120.5 / 480.0,
             omega_squared = (120.5 - 3 * 9.5) / (480.0 + 9.5),
             odds_ratio = (a * d) / (b * cc),
             risk_ratio = (a / (a + b)) / (cc / (cc + d)),
             effectsize_d_to_r_check = effectsize::d_to_r(0.8)))
})

local({
    n1 <- length(es1); n2 <- length(es2)
    ci  <- t.test(es1)$conf.int
    cid <- t.test(es1, es2)$conf.int
    cip <- t.test(es1, es2, var.equal = TRUE)$conf.int
    sig <- 0.2
    z   <- qnorm(0.975)
    dfv <- n1 - 1
    emit("EstimationMeans",
         "R: t.test(...)$conf.int, plus the z-interval and variance interval closed forms",
         data   = list(x = es1, y = es2, sigma = sig, confidence = 0.95),
         expect = list(
             ci_mean_lower = ci[1], ci_mean_upper = ci[2], ci_mean_estimate = mean(es1),
             ci_mean_z_lower = mean(es1) - z * sig / sqrt(n1),
             ci_mean_z_upper = mean(es1) + z * sig / sqrt(n1),
             ci_diff_welch_lower = cid[1], ci_diff_welch_upper = cid[2],
             ci_diff_pooled_lower = cip[1], ci_diff_pooled_upper = cip[2],
             ci_variance_lower = dfv * var(es1) / qchisq(0.975, dfv),
             ci_variance_upper = dfv * var(es1) / qchisq(0.025, dfv),
             standard_error = sd(es1) / sqrt(n1),
             margin_of_error_mean = qt(0.975, dfv) * sd(es1) / sqrt(n1)),
         prtol = 1e-9,
         caveat = "ci_mean_diff delegates to the pooled, equal-variance interval; ci_mean_diff_welch is the separate-variance form. Interval bounds and margins pass through statcpp's own t and normal quantile functions, which agree with R to about 1e-10, so they use the looser tier.")

    s <- 45L; nn <- 100L; s2 <- 30L; n2b <- 90L
    ph <- s / nn
    wl <- prop.test(s, nn, correct = FALSE)$conf.int
    p1 <- s / nn; p2 <- s2 / n2b
    emit("EstimationProportions",
         "R: prop.test(correct = FALSE)$conf.int for Wilson, plus the Wald closed forms",
         data   = list(successes = s, trials = nn, successes2 = s2, trials2 = n2b,
                       confidence = 0.95),
         expect = list(
             wald_lower = ph - z * sqrt(ph * (1 - ph) / nn),
             wald_upper = ph + z * sqrt(ph * (1 - ph) / nn),
             wilson_lower = wl[1], wilson_upper = wl[2],
             diff_lower = (p1 - p2) - z * sqrt(p1 * (1 - p1) / nn + p2 * (1 - p2) / n2b),
             diff_upper = (p1 - p2) + z * sqrt(p1 * (1 - p1) / nn + p2 * (1 - p2) / n2b),
             margin_of_error = z * sqrt(ph * (1 - ph) / nn),
             margin_of_error_worst_case = z * 0.5 / sqrt(nn),
             sample_size_mean = ceiling((z * 0.2 / 0.05)^2),
             sample_size_proportion = ceiling((z / 0.03)^2 * 0.5 * 0.5),
             moe_target = 0.05, moe_sigma = 0.2, moe_p_target = 0.03),
         prtol = 1e-8,
         caveat = "ci_proportion is the Wald interval; ci_proportion_wilson is the score interval and matches prop.test(correct = FALSE). Bounds pass through statcpp's own normal quantile, which agrees with R to about 2e-10.")
})

local({
    # statcpp documents that its t-test power uses a normal approximation rather
    # than the noncentral t distribution R uses, so the reference reproduces the
    # documented closed form and records R's value for comparison.
    pw <- function(d, n, alpha) {
        ncp <- d * sqrt(n)
        zc <- qnorm(1 - alpha / 2)
        1 - pnorm(zc - ncp) + pnorm(-zc - ncp)
    }
    d <- 0.5; n <- 30L; alpha <- 0.05
    pw2 <- function(d, n1, n2, alpha) {
        ncp <- d * sqrt(n1 * n2 / (n1 + n2))
        zc <- qnorm(1 - alpha / 2)
        1 - pnorm(zc - ncp) + pnorm(-zc - ncp)
    }
    # statcpp starts from the closed-form normal approximation and then increments
    # the sample size until the approximate power reaches the target.
    ss1 <- function(d, power, alpha) {
        za <- qnorm(1 - alpha / 2); zb <- qnorm(power)
        n <- max(2, ceiling((za + zb)^2 / abs(d)^2))
        for (i in 1:100) { if (pw(d, n, alpha) >= power) break; n <- n + 1 }
        n
    }
    ss2 <- function(d, power, alpha, ratio) {
        za <- qnorm(1 - alpha / 2); zb <- qnorm(power)
        n1 <- max(2, ceiling((za + zb)^2 / abs(d)^2 * (1 + 1 / ratio)))
        n2 <- max(2, ceiling(n1 * ratio))
        for (i in 1:100) {
            if (pw2(d, n1, n2, alpha) >= power) break
            n1 <- n1 + 1; n2 <- max(2, ceiling(n1 * ratio))
        }
        n1
    }
    emit("PowerTTest",
         "R: statcpp's documented normal approximation; power.t.test recorded for comparison",
         data   = list(effect_size = d, n = n, alpha = alpha, n1 = 25L, n2 = 35L,
                       target_power = 0.80, ratio = 1.0),
         expect = list(
             power_one_sample = pw(d, 30, alpha),
             power_two_sample = pw2(d, 25, 35, alpha),
             sample_size_one = as.double(ss1(d, 0.80, alpha)),
             sample_size_two = as.double(ss2(d, 0.80, alpha, 1.0)),
             r_power_t_test = power.t.test(n = 30, delta = d, sd = 1, sig.level = alpha,
                                           type = "one.sample")$power,
             r_sample_size = ceiling(power.t.test(delta = d, sd = 1, power = 0.80,
                                                 sig.level = alpha,
                                                 type = "one.sample")$n)),
         prtol = 1e-8,
         caveat = "power_one_sample is NOT expected to equal r_power_t_test: statcpp uses a normal approximation, R uses the noncentral t distribution. Both are recorded so the size of the gap is visible.")

    p1 <- 0.65; p2 <- 0.45; np <- 100L
    emit("PowerPropTest",
         "R: power.prop.test(n, p1, p2); statcpp implements the same closed form",
         data   = list(p1 = p1, p2 = p2, n = np, alpha = 0.05, target_power = 0.80),
         expect = list(
             power = power.prop.test(n = np, p1 = p1, p2 = p2, sig.level = 0.05)$power,
             sample_size = ceiling(power.prop.test(p1 = p1, p2 = p2, power = 0.80,
                                                  sig.level = 0.05)$n)))
})

write_header("r_reference_inference.hpp",
             "R reference values for effect sizes, interval estimation and power analysis.")

# ===========================================================================
# Phase 5a: applied analysis
# ===========================================================================

# ---------------------------------------------------------------------------
# survival.hpp
# ---------------------------------------------------------------------------

local({
    t1 <- c(6, 6, 6, 7, 10, 13, 16, 22, 23, 6, 9, 10, 11, 17, 19, 20, 25, 32, 32, 34)
    e1 <- c(1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    t2 <- c(1, 1, 2, 2, 3, 4, 4, 5, 5, 8, 8, 8, 8, 11, 11, 12, 12, 15, 17, 22)
    e2 <- c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

    fit <- survfit(Surv(t1, e1) ~ 1, conf.type = "plain")
    # statcpp prepends the origin, so the reference starts with t = 0, S = 1.
    ev  <- fit$n.event > 0
    emit("KaplanMeier", "R: survfit(Surv(time, event) ~ 1, conf.type = \"plain\"), with the origin prepended",
         data   = list(times = t1, events = as.double(e1)),
         expect = list(
             event_times = c(0, fit$time[ev]),
             survival = c(1, fit$surv[ev]),
             se = c(0, fit$std.err[ev] * fit$surv[ev]),
             n_at_risk = c(length(t1), as.double(fit$n.risk[ev])),
             n_events = c(0, as.double(fit$n.event[ev])),
             median = unname(summary(fit)$table["median"])),
         prtol = 1e-9,
         caveat = "R's survfit stores std.err on the log scale, so it is multiplied by the survival estimate to obtain the standard error of S(t) that statcpp reports. conf.type = \"plain\" matches the linear interval statcpp builds.")

    # statcpp accumulates d_i / n_i directly. R's type = "fh" applies the
    # Fleming-Harrington tie correction, sum_{j<d} 1/(n-j), which differs whenever
    # events are tied, so the reference uses the plain increments statcpp defines.
    ev2 <- fit$n.event > 0
    emit("NelsonAalen",
         "R: cumulative sum of d_i / n_i over the event times, with the origin prepended",
         data   = list(times = t1, events = as.double(e1)),
         expect = list(event_times = c(0, fit$time[ev2]),
                       cumulative_hazard = c(0, cumsum(fit$n.event[ev2] / fit$n.risk[ev2])),
                       fh_check = c(0, -log(survfit(Surv(t1, e1) ~ 1, type = "fh")$surv[ev2]))),
         prtol = 1e-9,
         caveat = "fh_check records R's Fleming-Harrington cumulative hazard. It differs from statcpp wherever events are tied: at t = 6 with three tied events out of twenty at risk, statcpp gives 3/20 = 0.15 while the tie-corrected form gives 0.15819.")

    tt <- c(t1, t2); ee <- c(e1, e2); gg <- c(rep(1, length(t1)), rep(2, length(t2)))
    sd <- survdiff(Surv(tt, ee) ~ gg)
    emit("LogrankTest", "R: survival::survdiff(Surv(time, event) ~ group)",
         data   = list(times1 = t1, events1 = as.double(e1),
                       times2 = t2, events2 = as.double(e2)),
         expect = list(statistic = unname(sd$chisq),
                       p_value = pchisq(unname(sd$chisq), 1, lower.tail = FALSE),
                       observed1 = as.double(sd$obs[1]), observed2 = as.double(sd$obs[2]),
                       expected1 = as.double(sd$exp[1]), expected2 = as.double(sd$exp[2])),
         prtol = 1e-7)
})

# ---------------------------------------------------------------------------
# time_series.hpp
# ---------------------------------------------------------------------------

local({
    ts <- c(12.4, 13.1, 11.8, 14.6, 15.2, 13.9, 16.1, 17.3, 15.8, 18.2,
            19.4, 17.9, 20.1, 21.5, 19.8, 22.3, 23.1, 21.6, 24.0, 25.2)
    pred <- c(12.0, 13.5, 12.2, 14.0, 15.5, 14.2, 15.8, 17.0, 16.2, 18.0,
              19.0, 18.3, 19.8, 21.0, 20.2, 22.0, 23.5, 22.0, 23.6, 25.0)
    m <- 8L
    a <- acf(ts, lag.max = m, plot = FALSE, demean = TRUE)$acf[, 1, 1]
    # statcpp reports PACF(0) = 1; R's pacf starts at lag 1.
    pa <- c(1, pacf(ts, lag.max = m, plot = FALSE)$acf[, 1, 1])
    w <- 5L
    emit("TimeSeries",
         "R: acf, pacf, diff, stats::filter for the moving average, and the error closed forms",
         data   = list(x = ts, y = pred, max_lag = m, window = w, period = 4L, alpha = 0.3,
                       lag_k = 3L),
         expect = list(
             acf = a,
             autocorrelation_lag3 = a[4],
             pacf = pa,
             diff1 = diff(ts, lag = 1, differences = 1),
             diff2 = diff(ts, lag = 1, differences = 2),
             seasonal_diff = diff(ts, lag = 4),
             lag3 = ts[seq_len(length(ts) - 3)],
             moving_average = as.numeric(na.omit(stats::filter(ts, rep(1 / w, w), sides = 1))),
             ema = Reduce(function(prev, x) 0.3 * x + 0.7 * prev, ts[-1], ts[1], accumulate = TRUE),
             mse = mean((ts - pred)^2),
             rmse = sqrt(mean((ts - pred)^2)),
             mae = mean(abs(ts - pred)),
             mape = mean(abs((ts - pred) / ts)) * 100),
         prtol = 1e-9,
         caveat = "statcpp's acf and pacf both start at lag 0; R's acf does too, while R's pacf starts at lag 1, so a leading 1 is prepended. lag3 is the series shifted by three, which statcpp returns without padding.")
})

# ---------------------------------------------------------------------------
# robust.hpp
# ---------------------------------------------------------------------------

local({
    rv <- c(2.1, 3.4, 2.8, 3.1, 2.9, 3.3, 2.7, 3.0, 3.2, 2.6, 9.8, 3.5, 2.4, 3.6, -4.2)
    med <- median(rv)
    q <- as.double(quantile(rv, c(0.25, 0.75), type = 7))
    iqrv <- q[2] - q[1]
    mad1 <- median(abs(rv - med))
    mzs <- 0.6745 * (rv - med) / mad1
    zs <- (rv - mean(rv)) / sd(rv)
    lo <- as.double(quantile(sort(rv), 0.05, type = 7))
    hi <- as.double(quantile(sort(rv), 0.95, type = 7))
    # Tukey biweight midvariance with c = 9
    u <- (rv - med) / (9 * mad1)
    keep <- abs(u) < 1
    # Tukey's biweight midvariance: the denominator is squared.
    num <- length(rv) * sum(((rv - med)^2 * (1 - u^2)^4)[keep])
    den <- sum(((1 - u^2) * (1 - 5 * u^2))[keep])^2
    wal <- outer(rv, rv, "+") / 2
    emit("Robust",
         "R: mad(constant = 1) and mad(), quantile-based fences, Walsh averages, Tukey biweight",
         data   = list(x = rv, k = 1.5, z_threshold = 3.0, mz_threshold = 3.5, limits = 0.05,
                       c = 9.0),
         expect = list(
             mad = mad1,
             mad_scaled = 1.4826 * mad1,
             mad_r_check = mad(rv),
             hodges_lehmann = median(wal[upper.tri(wal, diag = TRUE)]),
             hl_r_check = unname(wilcox.test(rv, conf.int = TRUE)$estimate),
             q1 = q[1], q3 = q[2], iqr_value = iqrv,
             lower_fence = q[1] - 1.5 * iqrv, upper_fence = q[2] + 1.5 * iqrv,
             n_outliers_iqr = as.double(sum(rv < q[1] - 1.5 * iqrv | rv > q[2] + 1.5 * iqrv)),
             n_outliers_z = as.double(sum(abs(zs) > 3.0)),
             n_outliers_mz = as.double(sum(abs(mzs) > 3.5)),
             winsorized = pmin(pmax(rv, lo), hi),
             biweight_midvariance = num / den),
         prtol = 1e-9,
         caveat = "mad_r_check confirms that statcpp's mad_scaled equals R's mad(). hl_r_check confirms that hodges_lehmann equals the pseudomedian wilcox.test reports.")

    res <- c(1.2, -0.8, 0.5, -1.5, 2.1, -0.3, 0.9, -1.1)
    hat <- c(0.30, 0.15, 0.22, 0.18, 0.35, 0.12, 0.25, 0.20)
    msev <- 1.4; pv <- 2L
    emit("Influence", "R: closed forms for Cook's distance and DFFITS",
         data   = list(residuals = res, hat_values = hat, mse = msev, p = pv),
         expect = list(
             cooks = res^2 * hat / (pv * msev * (1 - hat)^2),
             dffits = res * sqrt(hat) / (sqrt(msev) * (1 - hat))))
})

# ---------------------------------------------------------------------------
# multivariate.hpp
# ---------------------------------------------------------------------------

local({
    M <- matrix(c(4.2, 2.1, 5.5,  3.8, 3.4, 4.1,  5.1, 1.8, 6.2,  2.9, 4.6, 3.3,
                  6.3, 2.5, 7.1,  3.1, 3.9, 3.8,  5.8, 2.2, 6.5,  4.5, 3.1, 5.0,
                  2.6, 4.9, 2.9,  6.9, 1.5, 7.8,  3.6, 4.2, 4.0,  5.4, 2.8, 6.0),
                ncol = 3, byrow = TRUE)
    n <- nrow(M)
    # Sign of an eigenvector is arbitrary; both sides are normalised so that the
    # largest-magnitude loading of each component is positive.
    fix_sign <- function(v) if (v[which.max(abs(v))] < 0) -v else v
    pr <- prcomp(M, center = TRUE, scale. = FALSE)
    comps <- apply(pr$rotation[, 1:2, drop = FALSE], 2, fix_sign)
    ev <- pr$sdev^2
    scaled <- sweep(M, 2, apply(M, 2, min), "-")
    scaled <- sweep(scaled, 2, apply(M, 2, max) - apply(M, 2, min), "/")
    eg <- eigen(cov(M))
    emit("Multivariate",
         "R: cov, cor, scale, prcomp and eigen; component signs normalised on both sides",
         data   = list(x = M, n_components = 2L),
         expect = list(
             covariance_matrix = as.vector(t(cov(M))),
             correlation_matrix = as.vector(t(cor(M))),
             standardized = as.vector(t(scale(M))),
             min_max_scaled = as.vector(t(scaled)),
             components = as.vector(t(comps)),
             explained_variance = ev[1:2],
             explained_variance_ratio = (ev / sum(ev))[1:2],
             power_iteration_value = eg$values[1]),
         prtol = 1e-6,
         caveat = "statcpp obtains the components by power iteration with deflation rather than a full eigendecomposition, so agreement is about 1.8e-07 relative. PCA eigenvector signs are arbitrary, so each component is normalised to have a positive largest-magnitude loading before comparison. standardize uses the sample standard deviation, matching scale().")

    # pca_transform: scores on the sign-normalised components.
    centred <- sweep(M, 2, colMeans(M), "-")
    emit("PcaTransform", "R: centred data projected onto the sign-normalised components",
         data   = list(x = M, n_components = 2L),
         expect = list(scores = as.vector(t(centred %*% comps))),
         prtol = 1e-5,
         caveat = "Scores inherit the loading error from power iteration with deflation and amplify it: measured agreement is 1.8e-06 relative.")
})

# ---------------------------------------------------------------------------
# categorical.hpp
# ---------------------------------------------------------------------------

local({
    a <- 45L; b <- 25L; cc <- 20L; d <- 60L
    # statcpp hard-codes z = 1.96 for these intervals rather than using the exact
    # 0.975 normal quantile, so the reference uses the same constant.
    z <- 1.96
    lor <- log((a * d) / (b * cc)); se_lor <- sqrt(1/a + 1/b + 1/cc + 1/d)
    r1 <- a / (a + b); r2 <- cc / (cc + d)
    lrr <- log(r1 / r2); se_lrr <- sqrt(1/a - 1/(a + b) + 1/cc - 1/(cc + d))
    rd <- r1 - r2
    se_rd <- sqrt(r1 * (1 - r1) / (a + b) + r2 * (1 - r2) / (cc + d))
    rows <- c(rep(0L, a + b), rep(1L, cc + d))
    cols <- c(rep(0L, a), rep(1L, b), rep(0L, cc), rep(1L, d))
    tb <- table(rows, cols)
    emit("Categorical",
         "R: closed forms for the 2x2 measures, with table() for the contingency counts",
         data   = list(a = a, b = b, c = cc, d = d, rows = rows, cols = cols),
         expect = list(
             odds_ratio = (a * d) / (b * cc), log_odds_ratio = lor,
             se_log_odds_ratio = se_lor,
             or_ci_lower = exp(lor - z * se_lor), or_ci_upper = exp(lor + z * se_lor),
             relative_risk = r1 / r2, log_relative_risk = lrr,
             se_log_relative_risk = se_lrr,
             rr_ci_lower = exp(lrr - z * se_lrr), rr_ci_upper = exp(lrr + z * se_lrr),
             risk_difference = rd, se_risk_difference = se_rd,
             rd_ci_lower = rd - z * se_rd, rd_ci_upper = rd + z * se_rd,
             number_needed_to_treat = 1 / abs(rd),
             table_counts = as.double(as.vector(t(matrix(as.vector(tb), nrow = 2)))),
             row_totals = as.double(rowSums(tb)), col_totals = as.double(colSums(tb)),
             total = as.double(sum(tb)),
             z_exact = qnorm(0.975)),
         prtol = 1e-9,
         caveat = "The interval bounds use the hard-coded z = 1.96 that statcpp applies. z_exact records the exact 0.975 normal quantile, 1.959964; using it would move the bounds by about 1.3e-05 relative.")
})

# ---------------------------------------------------------------------------
# clustering.hpp
# ---------------------------------------------------------------------------

local({
    P <- matrix(c(1.0, 1.2,  1.3, 0.9,  0.8, 1.4,  5.2, 5.5,  5.6, 5.1,
                  4.9, 5.4,  9.1, 9.3,  9.4, 8.8,  8.9, 9.5,  9.2, 9.0),
                ncol = 2, byrow = TRUE)
    d <- dist(P, method = "euclidean")
    hc <- hclust(d, method = "single")
    labels <- cutree(hc, k = 3)
    sil <- cluster::silhouette(labels, d)
    a <- P[1, ]; b <- P[7, ]
    emit("Clustering",
         "R: dist, hclust(method = \"single\"), cutree and cluster::silhouette",
         data   = list(x = P, k = 3L, a = a, b = b),
         expect = list(
             euclidean_distance = sqrt(sum((a - b)^2)),
             manhattan_distance = sum(abs(a - b)),
             merge_heights = sort(hc$height),
             cluster_sizes = as.double(sort(as.vector(table(labels)))),
             silhouette_score = mean(sil[, "sil_width"])),
         prtol = 1e-9,
         caveat = "Merge heights are compared as a sorted vector and cluster sizes as sorted counts: statcpp numbers its dendrogram nodes and clusters differently from hclust, but a correct single-linkage tree has the same merge distances and the same partition.")
})

write_header("r_reference_applied.hpp",
             "R reference values for survival, time series, robust, multivariate, categorical and clustering.")

# ===========================================================================
# Phase 5b: data wrangling, missing data and numerical utilities
# ===========================================================================

local({
    dw <- c(4.0, 1.0, 9.0, 16.0, 25.0, 2.0, 36.0, 49.0, 3.0, 64.0)
    win <- 3L
    rmean <- sapply(win:length(dw), function(i) mean(dw[(i - win + 1):i]))
    rsum  <- sapply(win:length(dw), function(i) sum(dw[(i - win + 1):i]))
    rmin  <- sapply(win:length(dw), function(i) min(dw[(i - win + 1):i]))
    rmax  <- sapply(win:length(dw), function(i) max(dw[(i - win + 1):i]))
    rsd   <- sapply(win:length(dw), function(i) sd(dw[(i - win + 1):i]))
    lam <- 0.5
    emit("Transforms",
         "R: log, log1p, sqrt, the Box-Cox closed form, rank, and rolling windows",
         data   = list(x = dw, window = win, lambda = lam),
         expect = list(
             log_transform = log(dw),
             log1p_transform = log1p(dw),
             sqrt_transform = sqrt(dw),
             boxcox = (dw^lam - 1) / lam,
             rank_transform = rank(dw, ties.method = "average"),
             rolling_mean = rmean, rolling_sum = rsum,
             rolling_min = rmin, rolling_max = rmax, rolling_std = rsd),
         caveat = "The rolling functions return one value per complete window, so the result is shorter than the input by window - 1. rolling_std uses the sample standard deviation.")

    na_v <- c(1.0, NA, 3.0, NA, NA, 6.0, 7.0, NA, 9.0, 10.0)
    obs  <- na_v[!is.na(na_v)]
    ff <- na_v; for (i in seq_along(ff)) if (is.na(ff[i]) && i > 1) ff[i] <- ff[i - 1]
    bf <- na_v; for (i in rev(seq_along(bf))) if (is.na(bf[i]) && i < length(bf)) bf[i] <- bf[i + 1]
    emit("FillNa", "R: mean/median imputation, zoo::na.locf and zoo::na.approx",
         data   = list(x = na_v),
         expect = list(
             fill_mean = ifelse(is.na(na_v), mean(obs), na_v),
             fill_median = ifelse(is.na(na_v), median(obs), na_v),
             ffill = ff, bfill = bf,
             interpolate = as.numeric(zoo::na.approx(na_v, na.rm = FALSE))))

    cat_v <- c(3L, 7L, 3L, 5L, 9L, 3L, 7L, 1L, 5L, 7L, 3L, 2L)
    f <- factor(cat_v)
    tb <- table(cat_v)
    dupes <- sort(unique(cat_v[duplicated(cat_v)]))
    emit("Encoding",
         "R: sort, order, unique, duplicated, table, factor levels and model.matrix",
         data   = list(x = as.double(cat_v)),
         expect = list(
             sorted = as.double(sort(cat_v)),
             sorted_desc = as.double(sort(cat_v, decreasing = TRUE)),
             argsort = as.double(order(cat_v) - 1L),
             unique_values = as.double(unique(cat_v)),
             duplicated_values = as.double(dupes),
             value_counts_keys = as.double(names(tb)),
             value_counts_values = as.double(tb),
             label_encoded = as.double(match(cat_v, unique(cat_v)) - 1L),
             classes = as.double(unique(cat_v)),
             one_hot = as.vector(t(outer(cat_v, unique(cat_v), function(a, b) as.double(a == b))))),
         caveat = "argsort and label_encode are zero-based in C++ and one-based in R. label_encode assigns codes in order of first appearance, not by sorted level, so the reference uses match(x, unique(x)). get_duplicates returns its values in unspecified order, so both sides are sorted before comparison.")

    bd <- c(1.0, 2.5, 3.0, 4.5, 5.0, 6.5, 7.0, 8.5, 9.0, 10.5, 11.0, 12.5)
    nb <- 3L
    width_bins <- pmin(nb - 1L, floor((bd - min(bd)) / ((max(bd) - min(bd)) / nb)))
    emit("Binning", "R: equal-width bin index and the equal-frequency split",
         data   = list(x = bd, n_bins = nb),
         expect = list(equal_width = as.double(width_bins),
                       equal_freq_counts = as.double(as.vector(table(
                           cut(rank(bd, ties.method = "first"),
                               breaks = nb, labels = FALSE))))),
         caveat = "Both functions return a zero-based bin index per observation. equal_freq_counts records how many observations fall in each bin rather than the indices, because the tie-breaking rule at bin edges is implementation specific.")

    keys <- c("a", "b", "a", "c", "b", "a", "c", "b")
    vals <- c(1.5, 2.0, 3.5, 4.0, 5.5, 6.0, 7.5, 8.0)
    ks <- sort(unique(keys))
    emit("GroupBy", "R: tapply for the grouped sum, mean and count",
         data   = list(values = vals),
         expect = list(
             group_sum = as.double(tapply(vals, keys, sum)[ks]),
             group_mean = as.double(tapply(vals, keys, mean)[ks]),
             group_count = as.double(tapply(vals, keys, length)[ks])),
         caveat = "Keys are the strings a, b and c; statcpp returns them sorted, matching tapply's ordering.")
})

# ---------------------------------------------------------------------------
# missing_data.hpp
# ---------------------------------------------------------------------------

local({
    # The predictors must not be collinear, otherwise the imputation regression is
    # rank deficient and R drops a coefficient.
    M <- matrix(c(1.2,  4.8,  3.1,
                  4.5,  NA,   6.7,
                  7.3,  9.1,  2.4,
                  NA,   11.6, 12.9,
                  13.1, 8.4,  5.5,
                  16.8, 17.2, NA,
                  19.4, 12.7, 21.3,
                  22.6, 23.9, 8.8), ncol = 3, byrow = TRUE)
    cc <- complete.cases(M)
    # impute_conditional_mean uses only the FIRST entry of predictor_cols and fits
    # a simple linear regression, ignoring the remaining predictors.
    # The complete-case filter still requires every listed predictor to be present,
    # even though only the first one enters the regression.
    ok  <- !is.na(M[, 2]) & !is.na(M[, 1]) & !is.na(M[, 3])
    fit <- lm(M[ok, 2] ~ M[ok, 1])
    pred <- as.numeric(cbind(1, M[, 1]) %*% coef(fit))
    imputed <- ifelse(is.na(M[, 2]), pred, M[, 2])
    emit("MissingData",
         "R: complete.cases, is.na, cor(use = \"pairwise.complete.obs\") and lm-based imputation",
         data   = list(x = M),
         expect = list(
             complete_data = as.vector(t(M[cc, ])),
             n_complete = as.double(sum(cc)),
             n_dropped = as.double(sum(!cc)),
             proportion_complete = mean(cc),
             missing_indicator = as.vector(t(ifelse(is.na(M), 1, 0))),
             pairwise_correlation = as.vector(t(cor(M, use = "pairwise.complete.obs"))),
             imputed_column = imputed),
         prtol = 1e-9,
         caveat = "impute_conditional_mean takes a vector of predictor columns but uses only its first element, fitting a simple linear regression of the target on that one column. The complete-case filter still requires every listed predictor to be present. The reference reproduces that, not a multiple regression on every listed predictor.")
})

# ---------------------------------------------------------------------------
# numerical_utils.hpp
# ---------------------------------------------------------------------------

local({
    xs2 <- c(-0.9, -0.5, -1e-3, -1e-8, -1e-14, 0, 1e-14, 1e-8, 1e-3, 0.5, 1, 5, 20)
    # Naive left-to-right accumulation of 0.001 a thousand times drifts off 1 by
    # 7e-16; a compensated sum lands on 1 exactly.
    ks <- rep(0.001, 1000)
    emit("NumericalUtils", "R: log1p, expm1 and an exactly known compensated sum",
         data   = list(x = xs2, kahan_input = ks),
         expect = list(log1p = log1p(xs2), expm1 = expm1(xs2),
                       kahan_sum = 1.0,
                       naive_sum = sum(ks)),
         caveat = "Summing 0.001 a thousand times left to right gives 1.0000000000000007 in double precision; a compensated sum returns exactly 1. naive_sum records what R's sum() produces, which uses long-double accumulation and also lands on 1.")
})

write_header("r_reference_data.hpp",
             "R reference values for data wrangling, missing data and numerical utilities.")

# ===========================================================================
# Conditional functions: verifiable once glmnet, mice and naniar are available
# ===========================================================================

suppressPackageStartupMessages({ library(glmnet); library(mice); library(naniar) })

local({
    X <- matrix(c(2.1, 5.5,  3.4, 4.1,  1.8, 6.8,  5.2, 3.2,  4.6, 4.9,  6.9, 2.6,
                  3.3, 5.1,  7.1, 2.2,  5.8, 3.7,  8.4, 1.9,  6.2, 3.4,  9.1, 1.4),
                ncol = 2, byrow = TRUE)
    y <- c(13.1, 14.2, 11.8, 18.4, 18.0, 22.3, 14.9, 25.4, 19.6, 28.5, 22.2, 28.7)
    n <- nrow(X); p <- ncol(X)
    lam <- 5.0

    # glmnet minimises (1/2n) RSS + s (alpha |b|_1 + (1-alpha)/2 |b|_2^2), while
    # statcpp minimises (1/2) RSS + lambda |b|_1, so s = lambda / n.
    g <- glmnet(X, y, alpha = 1, lambda = lam / n, standardize = TRUE, intercept = TRUE,
                thresh = 1e-16)
    fitted_l <- as.numeric(coef(g)[1] + X %*% as.numeric(coef(g))[-1])
    emit("LassoRegression", "R: glmnet(alpha = 1, lambda = lambda / n, standardize = TRUE)",
         data   = list(x = X, y = y, lambda = lam),
         expect = list(coefficients = as.numeric(coef(g)),
                       mse = mean((y - fitted_l)^2)),
         prtol = 1e-5,
         caveat = "glmnet scales the residual sum of squares by 1/(2n) while statcpp scales it by 1/2, so the penalty maps as s = lambda / n. Both standardise predictors with the population standard deviation.")

    # The elastic net does NOT map onto glmnet the same way, so the reference is an
    # independent coordinate-descent solution of the objective statcpp defines.
    enet_cd <- function(X, y, lambda, alpha, iters = 100000) {
        nn <- nrow(X); pp <- ncol(X)
        xm <- colMeans(X)
        xs <- apply(X, 2, function(v) sqrt(sum((v - mean(v))^2) / length(v)))
        Z  <- sweep(sweep(X, 2, xm, "-"), 2, xs, "/")
        yc <- y - mean(y)
        l1 <- alpha * lambda; l2 <- (1 - alpha) * lambda
        b <- numeric(pp); r <- yc
        for (it in seq_len(iters)) {
            change <- 0
            for (j in seq_len(pp)) {
                r <- r + Z[, j] * b[j]
                xr <- sum(Z[, j] * r); xx <- sum(Z[, j]^2)
                bn <- sign(xr) * max(abs(xr) - l1, 0) / (xx + l2)
                change <- max(change, abs(bn - b[j]))
                b[j] <- bn
                r <- r - Z[, j] * b[j]
            }
            if (change < 1e-12) break
        }
        slopes <- b / xs
        c(mean(y) - sum(slopes * xm), slopes)
    }
    ce <- enet_cd(X, y, lam, 0.5)
    fitted_e <- ce[1] + as.numeric(X %*% ce[-1])
    emit("ElasticNetRegression",
         "R: independent coordinate descent on statcpp's objective, (1/2) RSS + a*lam |b|_1 + (1-a)*lam/2 |b|_2^2",
         data   = list(x = X, y = y, lambda = lam, alpha = 0.5),
         expect = list(coefficients = ce, mse = mean((y - fitted_e)^2),
                       glmnet_contrast = as.numeric(coef(glmnet(X, y, alpha = 0.5,
                           lambda = lam / n, standardize = TRUE, intercept = TRUE,
                           thresh = 1e-16)))),
         prtol = 1e-5,
         caveat = "Unlike the lasso, statcpp's elastic net does not coincide with glmnet under s = lambda / n: its ridge term is applied on a different scale. glmnet_contrast records what glmnet returns so the size of the gap is visible; it is NOT expected to match.")
})

local({
    M <- matrix(c(1.2,  4.8,  3.1,
                  4.5,  NA,   6.7,
                  7.3,  9.1,  NA,
                  NA,   11.6, 12.9,
                  13.1, 8.4,  5.5,
                  16.8, NA,   NA,
                  19.4, 12.7, 21.3,
                  22.6, 23.9, 8.8,
                  NA,   6.2,  14.1,
                  9.7,  15.3, 11.2), ncol = 3, byrow = TRUE)
    na_flag <- is.na(M)
    keys <- apply(na_flag, 1, function(r) paste(as.integer(r), collapse = ""))
    tb <- table(keys)
    emit("MissingPatterns",
         "R: colMeans(is.na), table of the missingness patterns, cross-checked with mice::md.pattern",
         data   = list(x = M),
         expect = list(
             missing_rates = unname(colMeans(na_flag)),
             overall_missing_rate = mean(na_flag),
             n_complete_cases = as.double(sum(complete.cases(M))),
             n_patterns = as.double(length(tb)),
             pattern_counts = as.double(sort(as.vector(tb), decreasing = TRUE)),
             mice_n_patterns = as.double(nrow(mice::md.pattern(M, plot = FALSE)) - 1)),
         caveat = "pattern_counts is compared as a sorted vector: statcpp and R enumerate the patterns in different orders. mice_n_patterns cross-checks the pattern count against mice::md.pattern.")

    # statcpp's test_mcar_simple accumulates a Welch t statistic for every ordered
    # pair of columns and converts the total with a Wilson-Hilferty approximation.
    mcar_simple <- function(M) {
        nc <- ncol(M); chi <- 0; df <- 0
        for (j in seq_len(nc)) {
            miss_j <- is.na(M[, j])
            for (k in seq_len(nc)) {
                if (j == k) next
                ok <- !is.na(M[, k])
                obs_k <- M[ok, k]; mj <- miss_j[ok]
                if (length(obs_k) < 5) next
                g1 <- obs_k[mj]; g2 <- obs_k[!mj]
                if (length(g1) < 2 || length(g2) < 2) next
                se <- sqrt(var(g1) / length(g1) + var(g2) / length(g2))
                if (se > 1e-10) { chi <- chi + ((mean(g1) - mean(g2)) / se)^2; df <- df + 1 }
            }
        }
        z <- (chi / df)^(1/3) - (1 - 2 / (9 * df))
        z <- z / sqrt(2 / (9 * df))
        c(chi = chi, df = df, p = min(1, max(0, 0.5 * erfc_(z / sqrt(2)))))
    }
    erfc_ <- function(x) 2 * pnorm(-x * sqrt(2))
    r <- mcar_simple(M)
    emit("McarSimple",
         "R: reproduction of statcpp's pairwise Welch accumulation with the Wilson-Hilferty p-value",
         data   = list(x = M),
         expect = list(chi_square = unname(r["chi"]), df = unname(r["df"]),
                       p_value = unname(r["p"]),
                       naniar_contrast = naniar::mcar_test(as.data.frame(M))$p.value),
         prtol = 1e-9,
         caveat = "statcpp's test_mcar_simple is not Little's MCAR test. naniar_contrast records what naniar::mcar_test reports for the same data; it is NOT expected to match, and the reference reproduces the heuristic statcpp actually implements.")
})

write_header("r_reference_conditional.hpp",
             "R reference values for functions that require glmnet, mice or naniar.")

# ---------------------------------------------------------------------------
# Cross-validation with deterministic folds
#
# Passing shuffle = false makes create_cv_folds split the data into contiguous
# blocks, so the whole procedure becomes reproducible.
# ---------------------------------------------------------------------------

local({
    X <- matrix(c(2.1, 5.5,  3.4, 4.1,  1.8, 6.8,  5.2, 3.2,  4.6, 4.9,  6.9, 2.6,
                  3.3, 5.1,  7.1, 2.2,  5.8, 3.7,  8.4, 1.9,  6.2, 3.4,  9.1, 1.4),
                ncol = 2, byrow = TRUE)
    y <- c(13.1, 14.2, 11.8, 18.4, 18.0, 22.3, 14.9, 25.4, 19.6, 28.5, 22.2, 28.7)
    n <- nrow(X); k <- 4L

    # create_cv_folds(n, k, shuffle = false): contiguous blocks, the first
    # (n mod k) folds taking one extra observation.
    make_folds <- function(n, k) {
        base <- n %/% k; rem <- n %% k; out <- list(); cur <- 1
        for (i in seq_len(k)) {
            sz <- base + if (i <= rem) 1 else 0
            out[[i]] <- cur:(cur + sz - 1); cur <- cur + sz
        }
        out
    }
    folds <- make_folds(n, k)

    fold_mse <- function(fit_fun) {
        vapply(folds, function(te) {
            tr <- setdiff(seq_len(n), te)
            pred <- fit_fun(X[tr, , drop = FALSE], y[tr], X[te, , drop = FALSE])
            mean((y[te] - pred)^2)
        }, numeric(1))
    }
    ols <- function(Xtr, ytr, Xte) {
        b <- coef(lm(ytr ~ Xtr))
        as.numeric(cbind(1, Xte) %*% b)
    }
    fe <- fold_mse(ols)
    emit("CrossValidateLinear",
         "R: contiguous k-fold split reproduced, with lm refitted on each training set",
         data   = list(x = X, y = y, k = k),
         expect = list(fold_errors = fe, mean_error = mean(fe),
                       se_error = sd(fe) / sqrt(k)),
         prtol = 1e-9,
         caveat = "Reproducible only because shuffle = false is passed; the default shuffles with the global random engine.")

    # Ridge and lasso paths over the same deterministic folds.
    ridge_fit <- function(lambda) function(Xtr, ytr, Xte) {
        xm <- colMeans(Xtr)
        xs <- apply(Xtr, 2, function(v) sqrt(sum((v - mean(v))^2) / length(v)))
        Z  <- sweep(sweep(Xtr, 2, xm, "-"), 2, xs, "/")
        b  <- as.numeric(solve(t(Z) %*% Z + lambda * diag(ncol(Xtr)), t(Z) %*% (ytr - mean(ytr))))
        sl <- b / xs
        as.numeric(mean(ytr) - sum(sl * xm) + Xte %*% sl)
    }
    grid <- c(0.1, 1.0, 5.0, 20.0)
    ridge_cv <- vapply(grid, function(l) mean(fold_mse(ridge_fit(l))), numeric(1))
    emit("CvRidge", "R: ridge closed form refitted on each contiguous fold",
         data   = list(x = X, y = y, k = k, lambda_grid = grid),
         expect = list(cv_errors = ridge_cv, best_lambda = grid[which.min(ridge_cv)]),
         prtol = 1e-4,
         caveat = "statcpp solves each ridge fit by coordinate descent with the API default tol = 1e-6, so the fold errors agree with the closed form to about that precision.")

    lasso_fit <- function(lambda) function(Xtr, ytr, Xte) {
        nn <- nrow(Xtr); pp <- ncol(Xtr)
        xm <- colMeans(Xtr)
        xs <- apply(Xtr, 2, function(v) sqrt(sum((v - mean(v))^2) / length(v)))
        Z  <- sweep(sweep(Xtr, 2, xm, "-"), 2, xs, "/")
        yc <- ytr - mean(ytr)
        b <- numeric(pp); r <- yc
        for (it in seq_len(100000)) {
            ch <- 0
            for (j in seq_len(pp)) {
                r <- r + Z[, j] * b[j]
                xr <- sum(Z[, j] * r); xx <- sum(Z[, j]^2)
                bn <- sign(xr) * max(abs(xr) - lambda, 0) / xx
                ch <- max(ch, abs(bn - b[j])); b[j] <- bn
                r <- r - Z[, j] * b[j]
            }
            if (ch < 1e-12) break
        }
        sl <- b / xs
        as.numeric(mean(ytr) - sum(sl * xm) + Xte %*% sl)
    }
    lasso_cv <- vapply(grid, function(l) mean(fold_mse(lasso_fit(l))), numeric(1))
    emit("CvLasso", "R: lasso coordinate descent refitted on each contiguous fold",
         data   = list(x = X, y = y, k = k, lambda_grid = grid),
         expect = list(cv_errors = lasso_cv, best_lambda = grid[which.min(lasso_cv)]),
         prtol = 1e-4)

    emit("CreateCvFolds",
         "R: contiguous k-fold index blocks, the first (n mod k) folds taking one extra observation",
         data   = list(n = as.integer(n), k = k),
         expect = list(fold_sizes = vapply(folds, length, numeric(1)),
                       first_indices = vapply(folds, function(f) as.double(f[1] - 1), numeric(1)),
                       flat_indices = as.double(unlist(folds) - 1)),
         caveat = "Indices are zero-based in C++ and one-based in R.")
})

write_header("r_reference_crossval.hpp",
             "R reference values for cross-validation with deterministic folds.")
