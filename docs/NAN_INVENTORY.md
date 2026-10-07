# NaN Handling Inventory (Appendix)

Appendix to [NAN_POLICY.md](NAN_POLICY.md). For every public function of v0.4.0,
one row shows its behaviour on input containing NaN and the behaviour under the
policy (measured 2026-10-07).

**v0.5.0 fixes all 191 functions whose verdict is `FIX`, `FIX-UB` or `FIX-HANG`.**
The "v0.4.0" columns are measurements of v0.4.0; the "Policy" column is the v0.5.0
implementation. Where the inventory listed several candidates, the column shows the
behaviour settled during implementation. `detect_outliers_zscore` was judged `OK` in the
inventory but was aligned with the other two outlier detectors in v0.5.0 and now throws.

## Legend

| Symbol | Meaning |
| --- | --- |
| `NaN` | every numeric output is NaN |
| `DROP` | same result as with NaN removed |
| `ELEM` | NaN only in the elements or windows depending on NaN |
| `CUM` | NaN from the position of NaN onwards |
| `VALUE` | a finite value that also differs from the result with NaN removed |
| `POSDEP` | result depends on the position of NaN |
| `THROW` | throws an exception |
| `HANG` | infinite loop |
| `UB` | undefined behaviour |

Verdict (as of v0.4.0): `OK` = already compliant, `FIX` = needed a fix,
`FIX-UB` / `FIX-HANG` = needed a fix for undefined behaviour / an infinite loop.
Classes are those of [NAN_POLICY.md](NAN_POLICY.md) section 4 (C1 to C7).

## Totals

| Verdict | Count |
| --- | --- |
| OK | 200 |
| FIX | 153 |
| FIX-UB | 22 |
| FIX-HANG | 16 |
| Total | 391 |

## anova.hpp

13 functions (OK 0, FIX 13, FIX-UB 0, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bonferroni_posthoc` | C6 | pairwise.t.test(y,g,p.adjust='bonferroni') | DROP | MIXED: stats NaN, p=1, significant=false | alpha:MIXED (CI NaN, significant=false, p ok) | DROP; alpha NaN THROW (#6) | FIX |
| `cohens_f` | C6 | effectsize::cohens_f(aov) | DROP | NaN (via one_way_anova) | - | DROP | FIX |
| `dunnett_posthoc` | C6 | closed form (Bonferroni t) | DROP (via aov) | MIXED: stats NaN, p=1, significant=false | alpha:MIXED (CI NaN, p ok) | DROP; alpha NaN THROW (#6) | FIX |
| `eta_squared[anova]` | C6 | effectsize::eta_squared(aov) | DROP | NaN (via one_way_anova) | - | DROP | FIX |
| `omega_squared[anova]` | C6 | effectsize::omega_squared(aov) | DROP | NaN (via one_way_anova) | - | DROP | FIX |
| `one_way_ancova` | C6 | car::Anova(lm(y~x+g),type=2) | DROP (row-wise, NA in y or x) | NaN | - | DROP row-wise | FIX |
| `one_way_anova` | C6 | summary(aov(y~g)) | DROP | NaN | - | DROP | FIX |
| `partial_eta_squared_a` | C6 | effectsize::eta_squared(aov,partial=TRUE) | DROP | NaN (via two_way_anova) | - | DROP (then unbalanced THROW) | FIX |
| `partial_eta_squared_b` | C6 | effectsize::eta_squared(aov,partial=TRUE) | DROP | NaN (via two_way_anova) | - | DROP (then unbalanced THROW) | FIX |
| `partial_eta_squared_interaction` | C6 | effectsize::eta_squared(aov,partial=TRUE) | DROP | NaN (via two_way_anova) | - | DROP (then unbalanced THROW) | FIX |
| `scheffe_posthoc` | C6 | closed form (qf) | DROP (via aov) | NaN (significant=false) | alpha:VALUE (finite CI; f_quantile(NaN)=1) | DROP; alpha NaN THROW (#6) | FIX |
| `tukey_hsd` | C6 | TukeyHSD(aov(y~g)) | DROP | MIXED: stats NaN, p=2.4e-14, significant=false | alpha:VALUE (finite CI; studentized_range_quantile(NaN)=1) | DROP; alpha NaN THROW (#6) | FIX |
| `two_way_anova` | C6 | summary(aov(y~a*b)) | DROP | NaN (NaN replacing a value); THROW unequal cell sizes (NaN added) | - | DROP, then the balanced-design check throws (#15) | FIX |

## basic_statistics.hpp

14 functions (OK 10, FIX 3, FIX-UB 1, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `argmax` | C1 | which.max(x)-1 | DROP (skips NA, index into original) | POSDEP mid:DROP first:VALUE(0 = NaN index) | - | DROP (index of max among non-NaN) | FIX |
| `argmin` | C1 | which.min(x)-1 | DROP (skips NA, index into original) | POSDEP mid:DROP first:VALUE(0 = NaN index) | - | DROP (index of min among non-NaN) | FIX |
| `count` | C1 | length(x) | VALUE (counts NA) | VALUE (counts NaN, 13) | - | VALUE (= length) | OK |
| `geometric_mean` | C1 | exp(mean(log(x))) | NA | NaN | - | NaN | OK |
| `harmonic_mean` | C1 | 1/mean(1/x) | NA | NaN | - | NaN | OK |
| `logarithmic_mean` | C1 | (b-a)/(log(b)-log(a)) | NA | NaN (a or b) | - | NaN | OK |
| `mean` | C1 | mean(x) | NA | NaN | - | NaN | OK |
| `median` | C1 | median(x) | NA | POSDEP mid:NaN first:VALUE(4.2) | - | NaN | FIX |
| `mode` | C3 | names(table(x))[which.max] | DROP | DROP | - | DROP | OK |
| `modes` | C3 | names(table(x))[tb==max] | DROP | DROP | - | DROP | OK |
| `sum` | C1 | sum(x) | NA | NaN | - | NaN | OK |
| `trimmed_mean` | C1 | mean(x, trim=0.2) | NA | POSDEP mid:NaN first:VALUE(4.667) | proportion=NaN: UB (float->size_t cast, basic_statistics.hpp:606); normal build VALUE 5.183 | data NaN; proportion NaN THROW (R errors) | FIX-UB |
| `weighted_harmonic_mean` | C1 | sum(w)/sum(w/x) | NA (x and w) | NaN (x and w) | - | NaN | OK |
| `weighted_mean` | C1 | weighted.mean(x, w) | NA (x and w) | NaN (x and w) | - | NaN | OK |

## categorical.hpp

5 functions (OK 5, FIX 0, FIX-UB 0, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `contingency_table` | C3 | table(r,c) | N/A(integer inputs) | N/A(integer inputs) | - | N/A | OK |
| `number_needed_to_treat` | C1 | closed form | N/A(integer inputs) | N/A(integer inputs) | - | N/A | OK |
| `odds_ratio[categorical]` | C1 | closed form (a*d)/(b*c) | N/A(integer inputs) | N/A(integer inputs) | - | N/A | OK |
| `relative_risk` | C1 | closed form | N/A(integer inputs) | N/A(integer inputs) | - | N/A | OK |
| `risk_difference` | C1 | closed form | N/A(integer inputs) | N/A(integer inputs) | - | N/A | OK |

## clustering.hpp

7 functions (OK 1, FIX 6, FIX-UB 0, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cut_dendrogram` | C3 | cutree(hc, k) | VALUE | distance field unused: NaN distance has no effect; dendrogram from NaN data gives a wrong partition (inherited) | n/a | n/a (integer inputs) | OK |
| `euclidean_distance` | C1 | sqrt(sum((a-b)^2)) (R ref); dist() | closed form NA; dist() rescales: VALUE 6.83 | NaN (a or b, any position) | n/a | R dist(): drop NaN coordinates, rescale sum by n/n_used (#3) | FIX |
| `hierarchical_clustering` | C3 | hclust(dist(X)) | VALUE (dist rescales NA cell; hclust on NA dist: ERROR) | MIXED: NaN point never merged; first n-2 merges equal the dropped data, last node is bogus self-merge (0,0) at DBL_MAX (ward 1.34e154) | n/a | THROW (clustering, like kmeans) (#12) | FIX |
| `kmeans` | C3 | kmeans(X, k) | ERROR: NA/NaN/Inf in foreign function call | MIXED: inertia NaN, a centroid NaN, labels collapse to cluster 0 | tol NaN -> VALUE (runs max_iter=100, same result) | THROW | FIX |
| `kmeans_plusplus_init` | C6 | none (random; kmeans errors) | n/a | VALUE: NaN row always chosen as a centroid (std::min keeps DBL_MAX, uniform_real_distribution(0, inf) -> threshold inf) | n/a | THROW (as kmeans) | FIX |
| `manhattan_distance` | C1 | sum(abs(a-b)) (R ref); dist(method="manhattan") | closed form NA; dist() rescales: VALUE 12 | NaN | n/a | R dist(): drop NaN coordinates, rescale sum by n/n_used (#3) | FIX |
| `silhouette_score` | C1 | mean(cluster::silhouette(labels, dist(X))[, 3]) | VALUE 0.919 (dist rescaling; clean 0.911) | NaN | n/a | THROW (clustering, like kmeans) (#12) | FIX |

## continuous_distributions.hpp

42 functions (OK 31, FIX 6, FIX-UB 0, FIX-HANG 5).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `beta_cdf` | C7 | pbeta(x,alpha,beta_param) | NA | NaN | NaN | NaN | OK |
| `beta_pdf` | C7 | dbeta(x,alpha,beta_param) | NA | NaN | NaN | NaN | OK |
| `beta_quantile` | C7 | qbeta(p,alpha,beta_param) | NA | NaN | NaN | NaN | OK |
| `beta_rand` | C7 | rbeta(1,alpha,beta_param) | n/a | n/a | HANG | NaN | FIX-HANG |
| `chisq_cdf` | C7 | pchisq(x,df) | NA | NaN | NaN | NaN | OK |
| `chisq_pdf` | C7 | dchisq(x,df) | NA | NaN | NaN | NaN | OK |
| `chisq_quantile` | C7 | qchisq(p,df) | NA | NaN | NaN | NaN | OK |
| `chisq_rand` | C7 | rchisq(1,df) | n/a | n/a | HANG | NaN | FIX-HANG |
| `exponential_cdf` | C7 | pexp(x,lambda) | NA | NaN | NaN | NaN | OK |
| `exponential_pdf` | C7 | dexp(x,lambda) | NA | NaN | NaN | NaN | OK |
| `exponential_quantile` | C7 | qexp(p,lambda) | NA | NaN | NaN | NaN | OK |
| `exponential_rand` | C7 | rexp(1,lambda) | n/a | n/a | NaN | NaN | OK |
| `f_cdf` | C7 | pf(x,df1,df2) | NA | NaN | NaN | NaN | OK |
| `f_pdf` | C7 | df(x,df1,df2) | NA | NaN | NaN | NaN | OK |
| `f_quantile` | C7 | qf(p,df1,df2) | NA | VALUE(1) | VALUE(1) | NaN | FIX |
| `f_rand` | C7 | rf(1,df1,df2) | n/a | n/a | HANG | NaN | FIX-HANG |
| `gamma_cdf` | C7 | pgamma(x,shape,rate) | NA | NaN | NaN | NaN | OK |
| `gamma_pdf` | C7 | dgamma(x,shape,rate) | NA | NaN | NaN | NaN | OK |
| `gamma_quantile` | C7 | qgamma(p,shape,rate) | NA | NaN | NaN | NaN | OK |
| `gamma_rand` | C7 | rgamma(1,shape,rate) | n/a | n/a | shape=HANG,rate=NaN | NaN | FIX-HANG |
| `lognormal_cdf` | C7 | plnorm(x,mu,sigma) | NA | NaN | NaN | NaN | OK |
| `lognormal_pdf` | C7 | dlnorm(x,mu,sigma) | NA | NaN | NaN | NaN | OK |
| `lognormal_quantile` | C7 | qlnorm(p,mu,sigma) | NA | NaN | NaN | NaN | OK |
| `lognormal_rand` | C7 | rlnorm(1,mu,sigma) | n/a | n/a | NaN | NaN | OK |
| `normal_cdf` | C7 | pnorm(x,mu,sigma) | NA | NaN | NaN | NaN | OK |
| `normal_pdf` | C7 | dnorm(x,mu,sigma) | NA | NaN | NaN | NaN | OK |
| `normal_quantile` | C7 | qnorm(p,mu,sigma) | NA | NaN | NaN | NaN | OK |
| `normal_rand` | C7 | rnorm(1,mu,sigma) | n/a | n/a | NaN | NaN | OK |
| `studentized_range_cdf` | C7 | ptukey(q,k,df) | NA | VALUE(0.999999) | k=VALUE(0.999999),df=VALUE(0) | NaN | FIX |
| `studentized_range_quantile` | C7 | qtukey(p,k,df) | NA | VALUE(1) | VALUE(1) | NaN | FIX |
| `t_cdf` | C7 | pt(x,df) | NA | NaN | NaN | NaN | OK |
| `t_pdf` | C7 | dt(x,df) | NA | NaN | NaN | NaN | OK |
| `t_quantile` | C7 | qt(p,df) | NA | NaN | VALUE(-0.52440) | NaN | FIX |
| `t_rand` | C7 | rt(1,df) | n/a | n/a | HANG | NaN | FIX-HANG |
| `uniform_cdf` | C7 | punif(x,a,b) | NA | NaN | NaN | NaN | OK |
| `uniform_pdf` | C7 | dunif(x,a,b) | NA | VALUE(1) | NaN | NaN | FIX |
| `uniform_quantile` | C7 | qunif(p,a,b) | NA | NaN | NaN | NaN | OK |
| `uniform_rand` | C7 | runif(1,a,b) | n/a | n/a | NaN | NaN | OK |
| `weibull_cdf` | C7 | pweibull(x,shape,scale) | NA | NaN | shape=POSDEP(x==scale->VALUE else NaN),scale=NaN | NaN | FIX |
| `weibull_pdf` | C7 | dweibull(x,shape,scale) | NA | NaN | NaN | NaN | OK |
| `weibull_quantile` | C7 | qweibull(p,shape,scale) | NA | NaN | NaN | NaN | OK |
| `weibull_rand` | C7 | rweibull(1,shape,scale) | n/a | n/a | NaN | NaN | OK |

## correlation_covariance.hpp

7 functions (OK 4, FIX 2, FIX-UB 0, FIX-HANG 1).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `covariance` | C1 | cov(x,y) | NA | NaN | NaN (mean_x/mean_y overload) | NaN | OK |
| `kendall_tau` | C1 | cor(x,y,method="kendall") | NA | VALUE | n/a | NaN | FIX |
| `pearson_correlation` | C1 | cor(x,y,method="pearson") | NA | NaN | NaN (mean_x/mean_y overload) | NaN | OK |
| `population_covariance` | C1 | cov(x,y)*(n-1)/n | NA | NaN | NaN (mean_x/mean_y overload) | NaN | OK |
| `sample_covariance` | C1 | cov(x,y) | NA | NaN | NaN (mean_x/mean_y overload) | NaN | OK |
| `spearman_correlation` | C1 | cor(x,y,method="spearman") | NA | HANG | n/a | NaN | FIX-HANG |
| `weighted_covariance` | C1 | cov.wt(cbind(x,y),wt,method="unbiased") | ERROR:'x' must contain finite values only | NaN | NaN (NaN weight passes w<0 check) | THROW | FIX |

## data_wrangling.hpp

39 functions (OK 20, FIX 17, FIX-UB 2, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `argsort` | C4 | order(x) | NaN index last | POSDEP: mid/first give wrong orders; 200-elem unsorted | - | NaN last (R order) | FIX |
| `bin_equal_freq` | C3 | cut(rank(x,ties='first'),n,labels=FALSE) | VALUE (NA ranked last, gets top bin) | POSDEP (argsort with NaN), no sanitizer report | n_bins size_t | THROW (policy §5) | FIX |
| `bin_equal_width` | C3 | cut(x,n,labels=FALSE) | ELEM NA | UB:float-cast-overflow nan->size_t at data_wrangling.hpp:1043; normal build mid VALUE bins, first all 0 | n_bins size_t | THROW (policy §5) | FIX-UB |
| `boxcox_transform` | C2 | (x^l-1)/l | ELEM | ELEM | lambda NaN: all NaN | ELEM / all NaN | OK |
| `drop_duplicates` | C3 | unique(x) | one NA kept | single NaN kept OK; {1,N,2,N,1} -> [1,nan,2,nan] (each NaN kept) | - | one NaN (R unique) | FIX |
| `dropna` | C5 | na.omit(x) / M[complete.cases(M),] | DROP | DROP (vec mid/first); matrix overload drops rows with NaN | - | DROP | OK |
| `fillna` | C5 | replace(x,is.na(x),v) | ELEM (filled) | ELEM filled | fill_value=NaN: NaN stays NaN | as R | OK |
| `fillna_bfill` | C5 | zoo::na.locf(x,fromLast=TRUE) | trailing NA DROPPED; interior filled | interior filled; trailing NaN kept (= na.rm=FALSE) | - | C5: keep length, trailing NaN stays NaN (deliberate difference #8) | OK |
| `fillna_ffill` | C5 | zoo::na.locf(x) | leading NA DROPPED (default na.rm=TRUE); interior filled | interior filled; leading NaN kept (= na.locf(na.rm=FALSE)) | - | C5: keep length, leading NaN stays NaN (deliberate difference #8) | OK |
| `fillna_interpolate` | C5 | zoo::na.approx(x) | leading/trailing NA DROPPED; interior linear | interior linear (matches 3.25 and repo ref); ends kept NaN (= na.rm=FALSE) | - | C5: keep length, edge NaN stays NaN (deliberate difference #8) | OK |
| `fillna_mean` | C5 | replace(x,is.na(x),mean(x,na.rm=TRUE)) | ELEM 3.9375 | ELEM 3.9375 mid/first | - | as R | OK |
| `fillna_median` | C5 | replace(x,is.na(x),median(x,na.rm=TRUE)) | ELEM 3.5 | VALUE 3.25 (mid/first); {9,N,1,5,3} filled 3 not 4 | - | median of observed (3.5) | FIX |
| `filter` | C2 | subset(x, x>2) [x[x>2] keeps NA] | subset: DROP; x[x>2]: ELEM NA kept | DROP (predicate false on NaN) | - | DROP (subset semantics; predicate is caller's) | OK |
| `filter_range` | C2 | subset(x, x>=a & x<=b) | DROP (subset); x[..] gives NA elems | DROP | min or max NaN: empty vector, silent | data DROP; NaN bound THROW (#9) | FIX |
| `filter_rows` | C2 | none | - | pass-through: rows with NaN kept if predicate true | - | caller predicate decides | OK |
| `get_duplicates` | C3 | unique(x[duplicated(x)]) | NA reported as duplicate | {1,N,2,N,1} -> [1] (NaN never duplicate) | - | NaN included once (R) | FIX |
| `group_by` | C3 | split(v,k) | value NA kept in group; key NA group DROPPED | value NaN kept in group; NaN double key silently MERGED into another group (std::map, NaN breaks ordering) | - | values pass-through; NaN key: row dropped (R split) | FIX |
| `group_count` | C3 | tapply(v,k,length) | value NA counted (3,2); key NA dropped (2,2) | value NaN counted (3,2) = R; NaN key merged -> VALUE (4,2) | - | value as R; NaN key: row dropped (R tapply) | FIX |
| `group_mean` | C2 | tapply(v,k,mean) | value NA: that group NA (ELEM); key NA: dropped (25,45) | value NaN: group NaN (ELEM, OK); NaN key merged -> VALUE (30,45) | - | value ELEM; NaN key: row dropped (R tapply) | FIX |
| `group_sum` | C2 | tapply(v,k,sum) | value NA: group NA; key NA dropped (50,90) | value NaN: group NaN OK; NaN key merged -> VALUE (120,90) | - | value ELEM; NaN key: row dropped (R tapply) | FIX |
| `is_na` | C5 | is.na(x) | TRUE | is_na(NaN)=true | - | true (R is.na) | OK |
| `label_encode` | C3 | match(x,unique(x)) / factor(x) | match: NA own code; factor: NA code | mid: NaN mapped to class 0 (code of 3); first: all codes 0, classes=[nan]; POSDEP | - | THROW (policy §5) | FIX |
| `log1p_transform` | C2 | log1p(x) | ELEM | ELEM | - | ELEM | OK |
| `log_transform` | C2 | log(x) | ELEM | ELEM | - | ELEM | OK |
| `one_hot_encode` | C3 | model.matrix(~factor(x)) | NA rows dropped (na.omit) | first: 1 column for all rows; mid: NaN row = class of 3; POSDEP | - | THROW (policy §5) | FIX |
| `rank_transform` | C4 | rank(x) | VALUE (NA gets rank n, na.last=TRUE) | ELEM (NaN kept, others ranked as NaN-removed) | - | keep NaN (policy §5, rank(na.last='keep')) | OK |
| `rolling_max` | C2 | zoo::rollmax(x,3) / rollapply max | ELEM | POSDEP ([4,4,4,5,nan,9,9]) | - | ELEM | FIX |
| `rolling_mean` | C2 | zoo::rollmean(x,3) | ELEM | ELEM | - | ELEM | OK |
| `rolling_min` | C2 | zoo::rollapply(x,3,min) | ELEM | POSDEP: NaN as 1st elem of window -> NaN, otherwise ignored ([1,1,1.5,1.5,nan,2,2]) | - | ELEM | FIX |
| `rolling_std` | C2 | zoo::rollapply(x,3,sd) | ELEM | ELEM | - | ELEM | OK |
| `rolling_sum` | C2 | zoo::rollsum(x,3) | ELEM | ELEM | - | ELEM | OK |
| `sample_with_replacement` | C2 | sample(x,replace=TRUE) | NA sampled like any value | NaN sampled like any value | - | pass-through | OK |
| `sample_without_replacement` | C2 | sample(x) | NA kept | NaN kept (permutation) | - | pass-through | OK |
| `sort_values` | C4 | sort(x) | DROP (both directions; NaN same) | POSDEP: mid -> unsorted garbage [1,1.5,3,4,nan,2,5,6,9]; first -> NaN kept at front; 200-elem non-NaN part unsorted | - | DROP (R sort) | FIX |
| `sqrt_transform` | C2 | sqrt(x) | ELEM | ELEM | - | ELEM | OK |
| `stratified_sample` | C2 | none | - | data NaN pass-through OK; NaN double strata silently merged into another stratum | sample_ratio=NaN passes (r<=0 or r>1) check -> UB:float-cast-overflow nan->size_t at data_wrangling.hpp:700 (normal build returns []) | data pass-through; NaN strata throw; ratio NaN throw | FIX-UB |
| `validate_data` | C5 | none | - | NaN counted as missing, is_valid=false (allow_missing=true -> true) | bool flags only | as documented | OK |
| `validate_range` | C5 | none | - | NaN data skipped (documented) | min or max NaN -> returns true (all comparisons false), silent | data skip; NaN bound THROW (#9) | FIX |
| `value_counts` | C3 | table(x) | DROP | mid: NaN counted into neighbour key (3:2); first: {nan:9} all values collapsed; POSDEP | - | THROW (policy §5) | FIX |

## discrete_distributions.hpp

31 functions (OK 19, FIX 8, FIX-UB 1, FIX-HANG 3).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bernoulli_cdf` | C7 | pbinom(k,1,p) | n/a | n/a | NaN | NaN | OK |
| `bernoulli_pmf` | C7 | dbinom(k,1,p) | n/a | n/a | NaN | NaN | OK |
| `bernoulli_quantile` | C7 | qbinom(prob,1,p) | NA | VALUE(1) | VALUE(1) | THROW (5: integer output) | FIX |
| `bernoulli_rand` | C7 | rbinom(1,1,p) | n/a | n/a | VALUE(0) | THROW (5: integer output) | FIX |
| `binomial_cdf` | C7 | pbinom(k,n,p) | n/a | n/a | NaN | NaN | OK |
| `binomial_coef` | C7 | choose(n,k) | n/a | n/a | n/a | n/a | OK |
| `binomial_pmf` | C7 | dbinom(k,n,p) | n/a | n/a | NaN | NaN | OK |
| `binomial_quantile` | C7 | qbinom(prob,n,p) | NA | VALUE(0) | VALUE(0) | THROW (5: integer output) | FIX |
| `binomial_rand` | C7 | rbinom(1,n,p) | n/a | n/a | VALUE(6505881574) | THROW (5: integer output) | FIX |
| `discrete_uniform_cdf` | C7 | none | n/a | n/a | n/a | n/a | OK |
| `discrete_uniform_pmf` | C7 | none | n/a | n/a | n/a | n/a | OK |
| `discrete_uniform_quantile` | C7 | none | n/a | UB:nan is outside the range of representable values | n/a | THROW (5: integer output) | FIX-UB |
| `discrete_uniform_rand` | C7 | none | n/a | n/a | n/a | n/a | OK |
| `geometric_cdf` | C7 | pgeom(k,p) | n/a | n/a | NaN | NaN | OK |
| `geometric_pmf` | C7 | dgeom(k,p) | n/a | n/a | NaN | NaN | OK |
| `geometric_quantile` | C7 | qgeom(prob,p) | NA | VALUE(0) | VALUE(0) | THROW (5: integer output) | FIX |
| `geometric_rand` | C7 | rgeom(1,p) | n/a | n/a | HANG | THROW (5: integer output) | FIX-HANG |
| `hypergeom_cdf` | C7 | phyper(k,K,N-K,n) | n/a | n/a | n/a | n/a | OK |
| `hypergeom_pmf` | C7 | dhyper(k,K,N-K,n) | n/a | n/a | n/a | n/a | OK |
| `hypergeom_quantile` | C7 | qhyper(p,K,N-K,n) | NA | VALUE(5) | n/a | THROW (5: integer output) | FIX |
| `hypergeom_rand` | C7 | rhyper(1,K,N-K,n) | n/a | n/a | n/a | n/a | OK |
| `log_binomial_coef` | C7 | lchoose(n,k) | n/a | n/a | n/a | n/a | OK |
| `log_factorial` | C7 | lfactorial(n) | n/a | n/a | n/a | n/a | OK |
| `nbinom_cdf` | C7 | pnbinom(k,r,p) | n/a | n/a | NaN | NaN | OK |
| `nbinom_pmf` | C7 | dnbinom(k,r,p) | n/a | n/a | NaN | NaN | OK |
| `nbinom_quantile` | C7 | qnbinom(prob,r,p) | NA | VALUE(0) | VALUE(0) | THROW (5: integer output) | FIX |
| `nbinom_rand` | C7 | rnbinom(1,r,p) | n/a | n/a | HANG | THROW (5: integer output) | FIX-HANG |
| `poisson_cdf` | C7 | ppois(k,lambda) | n/a | n/a | NaN | NaN | OK |
| `poisson_pmf` | C7 | dpois(k,lambda) | n/a | n/a | NaN | NaN | OK |
| `poisson_quantile` | C7 | qpois(prob,lambda) | NA | VALUE(0) | VALUE(0) | THROW (5: integer output) | FIX |
| `poisson_rand` | C7 | rpois(1,lambda) | n/a | n/a | HANG | THROW (5: integer output) | FIX-HANG |

## dispersion_spread.hpp

15 functions (OK 11, FIX 4, FIX-UB 0, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `coefficient_of_variation` | C1 | sd(x)/abs(mean(x)) | NA | NaN | precomputed_mean=NaN: NaN | NaN | OK |
| `geometric_stddev` | C1 | exp(sd(log(x))) | NA | NaN | - | NaN | OK |
| `iqr` | C1 | IQR(x, type=7) | ERROR (missing values and NaN's not allowed) | POSDEP mid:VALUE(3.4) first:VALUE(3.9) | - | THROW | FIX |
| `mean_absolute_deviation` | C1 | mean(abs(x-mean(x))) | NA | NaN | precomputed_mean=NaN: NaN | NaN | OK |
| `population_stddev` | C1 | sd(x)*sqrt((n-1)/n) | NA | NaN | precomputed_mean=NaN: NaN | NaN | OK |
| `population_variance` | C1 | var(x)*(n-1)/n | NA | NaN | precomputed_mean=NaN: NaN | NaN | OK |
| `range` | C1 | diff(range(x)) | NA | POSDEP mid:DROP first:NaN | - | NaN | FIX |
| `sample_stddev` | C1 | sd(x) | NA | NaN | precomputed_mean=NaN: NaN | NaN | OK |
| `sample_variance` | C1 | var(x) | NA | NaN | precomputed_mean=NaN: NaN | NaN | OK |
| `stddev` | C1 | sd(x) | NA | NaN | precomputed_mean=NaN: NaN | NaN | OK |
| `stdev` | C1 | sd(x)*sqrt((n-1)/n) (ddof=0) | NA | NaN | precomputed_mean=NaN: NaN | NaN | OK |
| `var` | C1 | var(x) (ddof=1) / closed form | NA | NaN | precomputed_mean=NaN: NaN | NaN | OK |
| `variance` | C1 | var(x) | NA | NaN | precomputed_mean=NaN: NaN | NaN | OK |
| `weighted_stddev` | C1 | sqrt(cov.wt(...)$cov) | ERROR (as weighted_variance) | NaN (x and w) | - | THROW (R cov.wt errors) (#17) | FIX |
| `weighted_variance` | C1 | cov.wt(cbind(x), wt=w, method=unbiased) | ERROR (x: 'x' must contain finite values only; w: missing value where TRUE/FALSE needed) | NaN (x and w) | - | THROW (R cov.wt errors) (#17) | FIX |

## distance_metrics.hpp

7 functions (OK 2, FIX 5, FIX-UB 0, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `chebyshev_distance` | C1 | dist(rbind(x,y),method="maximum") | DROP | DROP | n/a | R dist(): max over non-NaN coordinates (#3) | OK |
| `cosine_distance` | C1 | 1 - proxy::simil(method="cosine") | DROP (proxy) | NaN | n/a | pairwise DROP (proxy::simil) (#4) | FIX |
| `cosine_similarity` | C1 | proxy::simil(method="cosine") / closed form sum(x*y)/... | DROP (proxy drops pair) / closed form NA | NaN | n/a | pairwise DROP (proxy::simil) (#4) | FIX |
| `euclidean_distance` | C1 | dist(rbind(x,y),method="euclidean") | VALUE (pair dropped, sum rescaled by n/n_used: 13.59 vs 12.96) | NaN | n/a | R dist(): drop NaN coordinates, rescale sum by n/n_used (#3) | FIX |
| `mahalanobis_distance` | C1 | sqrt(mahalanobis(x,center,cov)) | NA | NaN | NaN (mean NaN, cov NaN: det NaN passes singular check) | NaN | OK |
| `manhattan_distance` | C1 | dist(rbind(x,y),method="manhattan") | VALUE (rescaled 37.4 vs 34) | NaN | n/a | R dist(): drop NaN coordinates, rescale sum by n/n_used (#3) | FIX |
| `minkowski_distance` | C1 | dist(rbind(x,y),method="minkowski",p=3) | VALUE (rescaled 10.24 vs 9.92) | NaN | NaN (p=NaN passes p<1 check) | R dist(): drop NaN coordinates, rescale (#3); p NaN THROW (#6) | FIX |

## effect_size.hpp

18 functions (OK 10, FIX 8, FIX-UB 0, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cohens_d` | C6 | effectsize::cohens_d(x,mu=) | DROP | NaN (both overloads) | mu0:NaN; sigma:NaN | DROP; mu0/sigma NaN THROW (#6) | FIX |
| `cohens_d_two_sample` | C6 | effectsize::cohens_d(x,y) | DROP | NaN | - | DROP | FIX |
| `cohens_h` | C7 | 2*asin(sqrt(p1))-2*asin(sqrt(p2)) | - | - | p1/p2:NaN | NaN | OK |
| `d_to_r` | C7 | effectsize::d_to_r(d) | - | - | d:NaN | NaN | OK |
| `eta_squared[effect_size]` | C7 | closed form ss_effect/ss_total | - | - | ss_effect/ss_total:NaN | NaN | OK |
| `glass_delta` | C6 | effectsize::glass_delta(trt,ctrl) | DROP | NaN (either group) | - | DROP | FIX |
| `hedges_correction_factor` | C7 | 1-3/(4*df-1) | - | - | df:NaN | NaN | OK |
| `hedges_g` | C6 | effectsize::hedges_g(x,mu=) | DROP | NaN | mu0:NaN | DROP; mu0 NaN THROW (#6) | FIX |
| `hedges_g_two_sample` | C6 | effectsize::hedges_g(x,y) | DROP | NaN | - | DROP | FIX |
| `interpret_cohens_d` | C3 | none (effectsize::interpret_cohens_d) | - | - | VALUE: large | THROW (§5 categorical output) | FIX |
| `interpret_correlation` | C3 | none (effectsize::interpret_r) | - | - | VALUE: large | THROW (§5 categorical output) | FIX |
| `interpret_eta_squared` | C3 | none (effectsize::interpret_eta_squared) | - | - | VALUE: large | THROW (§5 categorical output) | FIX |
| `odds_ratio[effect_size]` | C7 | closed form (a*d)/(b*c) | - | - | a/b/c/d:NaN | NaN | OK |
| `omega_squared[effect_size]` | C7 | closed form | - | - | all 4 params:NaN | NaN | OK |
| `partial_eta_squared` | C7 | closed form | - | - | f/df1/df2:NaN | NaN | OK |
| `r_to_d` | C7 | effectsize::r_to_d(r) | - | - | r:NaN | NaN | OK |
| `risk_ratio` | C7 | closed form | - | - | a/b/c/d:NaN | NaN | OK |
| `t_to_r` | C7 | effectsize::t_to_r(t,df) | - | - | t:NaN; df:NaN | NaN | OK |

## estimation.hpp

15 functions (OK 4, FIX 9, FIX-UB 2, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ci_mean` | C6 | t.test(x)$conf.int | DROP | NaN | MIXED: lo/up NaN, est finite | DROP / param THROW | FIX |
| `ci_mean_diff` | C6 | t.test(x,y,var.equal=TRUE)$conf.int | DROP (x or y) | NaN | MIXED: lo/up NaN, est finite | DROP / param THROW | FIX |
| `ci_mean_diff_pooled` | C6 | t.test(x,y,var.equal=TRUE)$conf.int | DROP (x or y) | NaN | MIXED: lo/up NaN, est finite | DROP / param THROW | FIX |
| `ci_mean_diff_welch` | C6 | t.test(x,y)$conf.int | DROP (x or y) | NaN | MIXED: lo/up NaN, est finite | DROP / param THROW | FIX |
| `ci_mean_z` | C6 | closed form mean(x)+-qnorm*sigma/sqrt(n) | NA | NaN | MIXED: lo/up NaN, est finite (sigma, conf) | DROP, like ci_mean / t.test (#5); confidence NaN THROW (#6) | FIX |
| `ci_proportion` | C6 | Wald closed form p+-qnorm*sqrt(p(1-p)/n) | n/a (size_t counts) | n/a | VALUE: [0,1] (std::max/min clamp swallows NaN) | confidence NaN THROW (#6) | FIX |
| `ci_proportion_diff` | C6 | closed form (Wald difference) | n/a (size_t counts) | n/a | MIXED: lo/up NaN, est finite | NaN bounds | OK |
| `ci_proportion_wilson` | C6 | prop.test(s,n,correct=FALSE)$conf.int | n/a (size_t counts) | n/a | MIXED: lo/up NaN, est finite | THROW | FIX |
| `ci_variance` | C6 | closed form (n-1)*var(x)/qchisq(...) | NA | NaN | MIXED: lo/up NaN, est finite | DROP, like ci_mean / t.test (#5); confidence NaN THROW (#6) | FIX |
| `margin_of_error_mean` | C1 | closed form qt*sd(x)/sqrt(n) (= t.test half width) | NA (closed form) / DROP (t.test) | NaN | NaN | DROP, like ci_mean / t.test (#5); confidence NaN THROW (#6) | FIX |
| `margin_of_error_proportion` | C1 | closed form qnorm*sqrt(p(1-p)/n) | n/a (size_t counts) | n/a | NaN | NaN | OK |
| `margin_of_error_proportion_worst_case` | C1 | closed form qnorm*0.5/sqrt(n) | n/a (size_t n) | n/a | NaN | NaN | OK |
| `sample_size_for_moe_mean` | C3 | closed form ceiling((qnorm*sigma/moe)^2) | n/a | n/a | UB: static_cast<size_t>(NaN) (UBSan float-cast-overflow) | THROW (integer output, policy 5) | FIX-UB |
| `sample_size_for_moe_proportion` | C3 | closed form ceiling((qnorm/moe)^2*p(1-p)) | n/a | n/a | UB: static_cast<size_t>(NaN) (garbage 5.3e11 or 0; UBSan float-cast-overflow) | THROW (integer output, policy 5) | FIX-UB |
| `standard_error` | C1 | sd(x)/sqrt(length(x)) | NA | NaN | NaN (precomputed_stddev overload) | NaN | OK |

## frequency_distribution.hpp

5 functions (OK 0, FIX 5, FIX-UB 0, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cumulative_frequency` | C3 | cumsum(table(x)) | DROP | POSDEP mid:VALUE (NaN merged into key 5) first:MIXED (single NaN key, 16) | - | THROW (C3, §5) ; R DROP | FIX |
| `cumulative_relative_frequency` | C3 | cumsum(prop.table(table(x))) | DROP | POSDEP mid:VALUE first:MIXED (single NaN key, 1.0) | - | THROW (C3, §5) ; R DROP | FIX |
| `frequency_count` | C3 | table(x) | DROP | VALUE (extra NaN key count 1 in unordered_map) | - | THROW (C3, §5) ; R DROP | FIX |
| `frequency_table` | C3 | table(x) / prop.table / cumsum | DROP (useNA=no) | POSDEP mid:NaN merged into key 5 (count 4), total 16; first: all values collapsed into one NaN key (count 16) | - | THROW (C3, §5 analogue of value_counts); R table drops NA | FIX |
| `relative_frequency` | C3 | prop.table(table(x)) | DROP | VALUE (NaN key 0.0625; denominators include NaN, 16) | - | THROW (C3, §5) ; R DROP | FIX |

## glm.hpp

12 functions (OK 6, FIX 6, FIX-UB 0, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `compute_glm_residuals` | C6 | residuals(glm, type=...) | DROP (na.omit: length n-1) | MIXED: response/pearson/working ELEM, but deviance residual at the NaN row is finite (-0 or -1.4e-05) | n/a | DROP | FIX |
| `glm_fit` | C6 | glm(y ~ x1 + x2) (gaussian default) | DROP | MIXED: gaussian y NaN -> all NaN but converged=true; X NaN -> NaN except null_deviance finite, converged=true; binomial y NaN -> coef NaN, se finite, deviance finite garbage | tol NaN -> VALUE (runs max_iter, same estimates, converged=false) | DROP; tol NaN THROW | FIX |
| `incidence_rate_ratios` | C2 | exp(coef(m))[-1] | ELEM | ELEM | n/a | ELEM | OK |
| `logistic_regression` | C6 | glm(family=binomial) | DROP | MIXED: y NaN passes [0,1] check -> coef NaN, se/deviance/aic finite garbage (322 vs 15), converged=true; X NaN -> coef/se NaN, deviance garbage, converged=true | tol NaN -> VALUE (same estimates, converged=false) | DROP; tol NaN THROW | FIX |
| `odds_ratios` | C2 | exp(coef(m))[-1] | ELEM | ELEM (coef NaN -> that OR NaN) | n/a | ELEM | OK |
| `odds_ratios_ci` | C2 | exp(confint.default(m))[-1, ] | ELEM | ELEM (coef or se NaN -> that pair NaN) | confidence NaN -> all NaN | ELEM; confidence NaN -> NaN | OK |
| `overdispersion_test` | C6 | sum(residuals(m,"pearson")^2)/df.residual | DROP | NaN | n/a | DROP | FIX |
| `poisson_regression` | C6 | glm(family=poisson) | DROP | MIXED: y NaN passes y>=0 check -> coef NaN, se finite, deviance finite garbage (8528), converged=true; X NaN -> coef/se NaN, deviance garbage | tol NaN -> VALUE (converged=false) | DROP; tol NaN THROW | FIX |
| `predict_count` | C7 | predict(glm, newdata, type="response") | NA | NaN | n/a | NaN | OK |
| `predict_probability` | C7 | predict(glm, newdata, type="response") | NA | NaN (x[0] or x[1] NaN) | n/a | NaN | OK |
| `pseudo_r_squared_mcfadden` | C1 | 1 - logLik(m)/logLik(m0) | DROP (via fit) | NaN (log_likelihood or null_ll NaN) | n/a | NaN (model-only input) | OK |
| `pseudo_r_squared_nagelkerke` | C1 | performance::r2_nagelkerke(m) | poisson: DROP; binomial: ERROR undefined columns selected (package bug) | VALUE: poisson y NaN silently skipped in saturated LL (y>0 false) while n unchanged; mid!=first | n/a (n is size_t) | DROP (remove NaN y and reduce n) | FIX |

## linear_regression.hpp

11 functions (OK 3, FIX 8, FIX-UB 0, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `adjusted_r_squared` | C1 | summary(lm)$adj.r.squared | DROP (lm); closed form NA | NaN (y or pred NaN) | n/a (num_predictors is size_t) | NaN (closed form from y and predictions) (#10) | FIX |
| `compute_residual_diagnostics` | C6 | residuals/rstandard/hatvalues/cooks.distance(lm) | DROP (na.omit: length n-1) | MIXED: y NaN -> residuals/studentized/cooks ELEM, durbin_watson=0; x/X NaN -> residuals ELEM, studentized all 0, hat & cooks all NaN, durbin_watson=0 | n/a | DROP | FIX |
| `compute_vif` | C6 | car::vif(lm) | DROP | NaN | n/a | DROP | FIX |
| `confidence_interval_mean` | C6 | predict(lm, interval="confidence") | DROP | MIXED: training x NaN -> prediction finite, lower/upper/se NaN | x_new NaN -> all NaN; confidence NaN -> prediction & se finite, bounds NaN | DROP (training x); confidence NaN THROW (#6); x_new NaN -> NaN | FIX |
| `correlation_matrix_determinant` | C1 | det(cor(X)) | NA | NaN | n/a | NaN | OK |
| `multicollinearity_score` | C1 | 1 - abs(det(cor(X))) | NA | NaN | n/a | NaN | OK |
| `multiple_linear_regression` | C6 | lm(y ~ x1 + x2) | DROP | NaN (y or X cell); df finite | n/a | DROP | FIX |
| `predict` | C7 | predict(lm, newdata) | NA | NaN (x / x[j] NaN, both overloads) | n/a | NaN | OK |
| `prediction_interval_simple` | C6 | predict(lm, interval="prediction") | DROP (fit from NA-x data equals clean) | MIXED: training x NaN -> prediction finite, lower/upper/se NaN | x_new NaN -> all NaN; confidence NaN -> prediction & se finite, bounds NaN | DROP (training x); confidence NaN THROW (#6); x_new NaN -> NaN | FIX |
| `r_squared` | C1 | summary(lm)$r.squared | DROP (lm); closed form 1-SSE/SST gives NA | NaN (y or pred NaN) | n/a | NaN (closed form from y and predictions) (#10) | FIX |
| `simple_linear_regression` | C6 | lm(y ~ x) | DROP | MIXED: y NaN -> coefs/t/p/R2/F NaN but ss_residual=residual_se=slope_se=intercept_se=0; x NaN -> NaN | n/a | DROP | FIX |

## missing_data.hpp

12 functions (OK 11, FIX 1, FIX-UB 0, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `analyze_missing_patterns` | C5 | colMeans(is.na(M)), complete.cases, mice::md.pattern | rates .2 .2 .2, overall .2, ncc 5, 5 patterns | identical to R (rates, overall, ncc=5, npat=5, counts {5,2,1,1,1}) | - | as R | OK |
| `correlation_matrix_pairwise` | C5 | cor(M,use='pairwise.complete.obs') | values .79439 .60318 .16966 | identical to R | - | as R | OK |
| `create_missing_indicator` | C5 | is.na(M)*1 | indicator | identical to R | - | as R | OK |
| `diagnose_missing_mechanism` | C5 | none | - | mcar on MP (via test_mcar_simple) | - | as documented | OK |
| `extract_complete_cases` | C5 | M[complete.cases(M),] | 5 rows | identical to R (n=5, dropped 5, prop .5) | - | as R | OK |
| `find_tipping_point` | C5 | none | - | NaN treated as missing; tp=-0.3375 found | threshold/delta_min/delta_max NaN -> tp NaN, found=false | NaN result | OK |
| `impute_conditional_mean` | C5 | lm on complete cases, first predictor | 7.136912, 15.454381 imputed | identical to R (preds {0,2}) | target_col/predictor_cols are size_t | as R | OK |
| `multiple_imputation_bootstrap` | C5 | none (random) | - | finite pooled stats; all-NaN column silently imputed with 0 (pooled_mean 0) | m,seed integral | all-NaN column -> NaN (like pmm) | FIX |
| `multiple_imputation_pmm` | C5 | mice (method pmm) - not comparable (random) | - | finite pooled stats; all-NaN column -> pooled_mean NaN | m,seed integral | as documented | OK |
| `sensitivity_analysis_pattern_mixture` | C5 | none | - | NaN treated as missing (designed); orig mean 3.9375 | delta NaN -> that estimate NaN (ELEM) | ELEM | OK |
| `sensitivity_analysis_selection_model` | C5 | none | - | NaN treated as missing (designed) | phi NaN -> that estimate NaN (ELEM) | ELEM | OK |
| `test_mcar_simple` | C5 | R reproduction of statcpp heuristic (naniar::mcar_test contrast) | chi 3.1392 df 4 p 0.53726 (naniar p 0.746) | identical to R reproduction; all-NaN column skipped silently | - | as R reproduction | OK |

## model_selection.hpp

15 functions (OK 6, FIX 9, FIX-UB 0, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `aic` | C1 | -2*logLik + 2k | NA | n/a | log_likelihood NaN -> NaN | NaN | OK |
| `aic_linear` | C1 | AIC(lm) | DROP (via fit) | ss_residual NaN -> NaN (both overloads) | n/a | NaN (model-only input) | OK |
| `aicc` | C1 | AIC + 2k(k+1)/(n-k-1) | NA | n/a | log_likelihood NaN -> NaN | NaN | OK |
| `bic` | C1 | -2*logLik + k*log(n) | NA | n/a | log_likelihood NaN -> NaN | NaN | OK |
| `bic_linear` | C1 | BIC(lm) | DROP (via fit) | ss_residual NaN -> NaN (both overloads) | n/a | NaN (model-only input) | OK |
| `create_cv_folds` | C3 | none (index helper) | n/a | n/a (no floating inputs) | n/a | n/a | OK |
| `cross_validate_linear` | C6 | none (lm per contiguous fold, reproduced) | reproduction: fold containing NA -> NA, mean NA (lm drops NA in training) | NaN (all fold errors, mean, se) | n/a | DROP (C6) | FIX |
| `cv_lasso` | C6 | none (lasso per fold, reproduced) | n/a (cv.glmnet NA x: ERROR) | MIXED: cv errors NaN, best_lambda=grid[0]=0.1 (clean best 1.0) | grid NaN -> VALUE (that lambda fits intercept-only, error 48.2) | THROW (glmnet); NaN grid entry THROW | FIX |
| `cv_ridge` | C6 | none (ridge per fold, reproduced) | n/a (cv.glmnet NA x: ERROR) | MIXED: all cv errors NaN but best_lambda=grid[0] (finite) | grid[0] NaN -> best_lambda NaN; grid[2] NaN -> best 0.1, cv_errors[2] NaN (POSDEP) | THROW (glmnet); NaN grid entry THROW | FIX |
| `elastic_net_regression` | C6 | none (own objective; glmnet contrast) | glmnet: ERROR | MIXED: intercept & mse NaN, slopes 0 (y NaN) or NaN/0 (X NaN) | lambda NaN -> NaN; alpha NaN -> NaN; tol NaN -> VALUE | THROW (glmnet); lambda/alpha/tol NaN THROW | FIX |
| `generate_lambda_grid` | C6 | none (statcpp-specific; glmnet lambda path) | glmnet: ERROR | y NaN -> THROW: lambda_max must be positive (data may be constant) [misleading]; X NaN -> VALUE (NaN column ignored by std::max, 7.95 vs 13.08) | lambda_min_ratio NaN -> all NaN (incl. lambda_max) | THROW (data contains NaN) | FIX |
| `lasso_regression` | C6 | glmnet(alpha=1, lambda=lambda/n) | ERROR (x: x has missing values; y: missing value where TRUE/FALSE needed) | MIXED: y NaN -> intercept & mse NaN, slopes 0; X NaN -> NaN/0 mix | lambda NaN -> VALUE (silently intercept-only fit, mse 30.6); tol NaN -> VALUE | THROW (R glmnet) | FIX |
| `loocv_linear` | C6 | mean((resid/(1-hat))^2) of lm | DROP | NaN | n/a | DROP | FIX |
| `press_statistic` | C6 | sum((resid(m)/(1-hatvalues(m)))^2) | DROP | NaN (x or y NaN) | n/a | DROP | FIX |
| `ridge_regression` | C6 | none (closed form); glmnet(alpha=0) errors on NA x | ERROR (glmnet: x has missing values) | NaN (y or X NaN) | lambda NaN -> NaN; tol NaN -> VALUE (runs max_iter) | THROW (glmnet); lambda/tol NaN THROW | FIX |

## multivariate.hpp

7 functions (OK 2, FIX 5, FIX-UB 0, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `correlation_matrix` | C2 | cor(X) | ELEM, diagonal stays 1 | ELEM but diagonal of NaN variable is NaN (R: 1) | n/a | ELEM with diagonal 1 | FIX |
| `covariance_matrix` | C2 | cov(X) | ELEM (row/col of NA variable NA) | ELEM (same cells as R) | n/a | ELEM | OK |
| `min_max_scale` | C2 | closed form (x-min)/(max-min) with min/max | whole column NA (min/max return NA) | NaN only at that cell; min/max ignore NaN via std::min/std::max (not POSDEP) | n/a | R scale()-like: column min/max from non-NaN, NaN only in that cell (#11) | OK |
| `pca` | C6 | prcomp(X) | ERROR: infinite or missing values in 'x' | NaN (all components/variances) | n/a (n_components size_t) | THROW | FIX |
| `pca_transform` | C6 | prcomp(X)$x | ERROR: infinite or missing values in 'x' | NaN: every row NaN (column mean NaN pollutes all scores) | n/a | THROW | FIX |
| `power_iteration` | C6 | eigen(m)$values[1] | ERROR: infinite or missing values in 'x' | NaN (eigenvalue and vector) | tol NaN -> VALUE (runs max_iter) | THROW | FIX |
| `standardize` | C2 | scale(X) | center/scale use na.rm; only the NA cell is NA | whole column of NaN variable NaN | n/a | R scale(): column stats from non-NaN, NaN only in that cell (#11) | FIX |

## nonparametric_tests.hpp

11 functions (OK 1, FIX 5, FIX-UB 0, FIX-HANG 5).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bartlett_test` | C6 | bartlett.test(groups) | DROP | NaN | - | DROP | FIX |
| `compute_ranks_with_ties` | C2 | rank(x,ties.method='average') | VALUE: NA gets largest rank (na.last=TRUE) | HANG | - | ELEM: NaN kept, like rank_transform (#14) | FIX-HANG |
| `compute_tie_groups` | C3 | none (helper) | - | HANG (NaN anywhere, incl. last) | - | THROW (size_t output) | FIX-HANG |
| `fisher_exact_test` | C6 | fisher.test(matrix) | N/A(integer inputs) | N/A(integer inputs) | - | N/A | OK |
| `kruskal_wallis_test` | C6 | kruskal.test(groups) | DROP | HANG | - | DROP | FIX-HANG |
| `ks_test_normal` | C6 | nortest::lillie.test(x) | DROP | VALUE: D=0,p=1 | - | DROP | FIX |
| `levene_test` | C6 | car::leveneTest(y,g,center=median) | DROP | NaN | - | DROP | FIX |
| `lilliefors_test` | C6 | nortest::lillie.test(x) | DROP | VALUE: D=0,p=1 | - | DROP | FIX |
| `mann_whitney_u_test` | C6 | wilcox.test(x,y) | DROP (NaN also DROP) | HANG | - | DROP | FIX-HANG |
| `shapiro_wilk_test` | C6 | shapiro.test(x) | DROP | VALUE: W=1,p=1 | - | DROP | FIX |
| `wilcoxon_signed_rank_test` | C6 | wilcox.test(x,mu=) | DROP | HANG | mu0:HANG | DROP; mu0 THROW | FIX-HANG |

## numerical_utils.hpp

15 functions (OK 14, FIX 1, FIX-UB 0, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `all_finite` | C7 | all(is.finite(x)) | FALSE | false | - | false | OK |
| `approx_equal` | C7 | none (all.equal not equivalent) | - | NaN vs anything -> false (incl NaN,NaN) | tol NaN -> only exact equality true | IEEE false | OK |
| `approx_equal_range` | C7 | none (all.equal) | - | identical ranges with NaN -> false | tol NaN -> only exact equality | false (consistent with approx_equal) | OK |
| `clamp` | C7 | pmax(lo,pmin(x,hi)) | NA | x=NaN -> VALUE 0 (returns min_val) | min NaN -> NaN; max NaN -> x returned unclamped (clamp(5,0,NaN)=5): MIXED | NaN | FIX |
| `expm1_safe` | C7 | expm1(x) | NA | NaN | - | NaN | OK |
| `has_converged` | C7 | none | - | false | abs_tol/rel_tol NaN -> false | false | OK |
| `has_converged_abs` | C7 | none | - | false | tol NaN -> false | false (never converged) | OK |
| `has_converged_rel` | C7 | none | - | false | tol NaN -> false | false | OK |
| `in_range` | C7 | none | - | false | bound NaN -> false | false | OK |
| `is_finite` | C7 | is.finite(x) | FALSE | false | - | false | OK |
| `is_zero` | C7 | none | - | false | tol NaN -> false | false | OK |
| `kahan_sum` | C1 | sum(x) | NA | NaN (both overloads, mid/first) | - | NaN | OK |
| `log1p_safe` | C7 | log1p(x) | NA | NaN | - | NaN | OK |
| `relative_error` | C7 | none | - | NaN | x_ref NaN -> NaN | NaN | OK |
| `safe_divide` | C7 | none | - | numerator NaN -> NaN | denominator NaN -> NaN; default_value NaN only used when denom~0 | NaN | OK |

## order_statistics.hpp

8 functions (OK 0, FIX 4, FIX-UB 4, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `five_number_summary` | C1 | quantile(x, c(0,.25,.5,.75,1), type=7) | ERROR (missing values and NaN's not allowed) | POSDEP mid:MIXED(median NaN) first:MIXED(min NaN) | - | THROW | FIX |
| `interpolate_at` | C1 | none (helper of percentile, inventory 3.4) | - | POSDEP mid:VALUE(3.66) first:VALUE(3.1) | p=NaN: UB (float->size_t cast, order_statistics.hpp:75/105); normal build HANG/garbage output (non-proj) | data THROW (quantile family); p NaN THROW (#6) | FIX-UB |
| `maximum` | C1 | max(x) | NA | POSDEP mid:DROP first:NaN | - | NaN | FIX |
| `minimum` | C1 | min(x) | NA | POSDEP mid:DROP first:NaN | - | NaN | FIX |
| `percentile` | C1 | quantile(x, p, type=7) | ERROR (missing values and NaN's not allowed) | POSDEP mid:VALUE(3.66) first:VALUE(3.1) | p=NaN: UB (float->size_t cast); normal build non-proj HANG (alarm) with garbage output, proj VALUE 1.7 | data THROW (R quantile errors); p NaN THROW (#6) | FIX-UB |
| `quartiles` | C1 | quantile(x, c(.25,.5,.75), type=7) | ERROR (missing values and NaN's not allowed) | POSDEP mid:MIXED(q2 NaN) first:VALUE | - | THROW | FIX |
| `weighted_median` | C1 | quantile(x, .5, type=2) at unit weights / closed form | ERROR (quantile); closed form: x NA VALUE (order puts NA last), w NA ERROR | x: POSDEP VALUE(2.1)/VALUE(4.2) + UB; w: VALUE(9.6) both | - | THROW (quantile family) | FIX-UB |
| `weighted_percentile` | C1 | quantile(x, p, type=2) at unit weights / closed form | ERROR (quantile); closed form: x NA VALUE, w NA ERROR | x: POSDEP VALUE(6.1)/VALUE(2.1) + UB; w: VALUE(9.6) both | p=NaN: THROW (p must be in [0,1]) | data THROW (quantile family); p NaN THROW (#6) | FIX-UB |

## parametric_tests.hpp

14 functions (OK 1, FIX 13, FIX-UB 0, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `benjamini_hochberg_correction` | C2 | p.adjust(p,'BH') | ELEM (NA kept; n=#non-NA) | POSDEP: NaN kept, other values differ mid vs first, none match R | - | ELEM with n=#non-NaN | FIX |
| `bonferroni_correction` | C2 | p.adjust(p,'bonferroni') | ELEM (NA kept; n=#non-NA) | MIXED: NaN->1.0 (std::min), n counts NaN | - | ELEM with n=#non-NaN | FIX |
| `chisq_test_gof` | C6 | chisq.test(obs,p=) | ERROR:all entries of 'x' must be nonnegative and finite | NaN | expected NaN: NaN (e<=0 check passes NaN) | THROW (data and expected) | FIX |
| `chisq_test_gof_uniform` | C6 | chisq.test(obs) | ERROR:all entries of 'x' must be nonnegative and finite | NaN | - | THROW | FIX |
| `chisq_test_independence` | C6 | chisq.test(m,correct=FALSE) | ERROR:all entries of 'x' must be nonnegative and finite | VALUE (chi2=0,p=1 silently) | - | THROW | FIX |
| `f_test` | C6 | var.test(x,y) | DROP | NaN | - | DROP | FIX |
| `holm_correction` | C2 | p.adjust(p,'holm') | ELEM (NA kept; n=#non-NA) | POSDEP: mid=NaN kept but neighbours wrong; first=NaN + others equal NaN-removed result | - | ELEM with n=#non-NaN | FIX |
| `t_test` | C6 | t.test(x,mu=) | DROP (NaN also DROP) | NaN (df counts NaN) | mu0:NaN | DROP; mu0 THROW | FIX |
| `t_test_paired` | C6 | t.test(x,y,paired=TRUE) | DROP (pairwise) | NaN | - | DROP pairwise | FIX |
| `t_test_two_sample` | C6 | t.test(x,y,var.equal=TRUE) | DROP | NaN | - | DROP | FIX |
| `t_test_welch` | C6 | t.test(x,y) | DROP | NaN | - | DROP | FIX |
| `z_test` | C6 | none (closed form pnorm) | - | NaN | mu0:NaN; sigma:NaN | DROP; params THROW (analogy t.test mu=NA error) | FIX |
| `z_test_proportion` | C6 | prop.test(x,n,p,correct=FALSE) | N/A(integer inputs) | N/A(integer inputs) | p0:NaN | param THROW | FIX |
| `z_test_proportion_two_sample` | C6 | prop.test(c(x1,x2),c(n1,n2)) | N/A(integer inputs) | N/A(integer inputs) | - | N/A | OK |

## power_analysis.hpp

8 functions (OK 0, FIX 8, FIX-UB 0, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `power_analysis_t_one_sample` | C7 | closed form | - | - | effect:power NaN; alpha:power NaN | effect NaN; alpha THROW | FIX |
| `power_analysis_t_one_sample_n` | C3 | closed form + search | - | - | effect/alpha: sample_size 102; power: power NaN,size 102 | THROW | FIX |
| `power_prop_test` | C7 | power.prop.test(n,p1,p2) | - | - | p1/p2:NaN; alpha:NaN | p NaN; alpha THROW | FIX |
| `power_t_test_one_sample` | C7 | closed form (normal approx; cf power.t.test) | - | - | effect_size:NaN; alpha:NaN | effect NaN; alpha THROW (R sig.level error) | FIX |
| `power_t_test_two_sample` | C7 | closed form (normal approx) | - | - | effect_size:NaN; alpha:NaN | effect NaN; alpha THROW | FIX |
| `sample_size_prop_test` | C3 | power.prop.test(p1,p2,power) | - | - | p1/p2/power/alpha: VALUE 102 | THROW | FIX |
| `sample_size_t_test_one_sample` | C3 | closed form + search (cf power.t.test) | - | - | effect/power/alpha: VALUE 102 silently | THROW | FIX |
| `sample_size_t_test_two_sample` | C3 | closed form + search | - | - | effect/power/alpha/ratio: VALUE 102 | THROW | FIX |

## random_engine.hpp

3 functions (OK 3, FIX 0, FIX-UB 0, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `get_random_engine` | n/a | none | - | no floating input | - | - | OK |
| `randomize_seed` | n/a | set.seed(NULL) | - | no input | - | - | OK |
| `set_seed` | n/a | set.seed(seed) | - | uint64 param; NaN not representable (caller cast would be caller UB) | - | - | OK |

## resampling.hpp

9 functions (OK 1, FIX 3, FIX-UB 5, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bootstrap` | C6 | none (random) | n/a | MIXED: est/se/bias NaN, ci bounds partly finite (sort over NaN replicates) | UB: confidence NaN -> static_cast<size_t>(floor(NaN)) index | data passed to stat_func unchanged, as R boot (#16); confidence NaN THROW (#6) | FIX-UB |
| `bootstrap_bca` | C6 | none (random) | n/a | UB: NaN data -> alpha1/alpha2 NaN -> static_cast<size_t> in clamp_index | UB: confidence NaN index cast | data passed to the statistic unchanged, as R's boot (#16); confidence NaN THROW (#6) | FIX-UB |
| `bootstrap_mean` | C6 | none (random) | n/a | MIXED: est NaN, ci_lower finite, ci_upper NaN or finite depending on position | UB: confidence NaN index cast | DROP / param THROW | FIX-UB |
| `bootstrap_median` | C6 | none (random) | n/a | POSDEP: mid est NaN, first est 4 (median of sorted vector with NaN), se NaN | UB: confidence NaN index cast | DROP / param THROW | FIX-UB |
| `bootstrap_sample` | C2 | none (random) | n/a | ELEM (NaN copied into resample where drawn) | n/a | ELEM (pass-through) | OK |
| `bootstrap_stddev` | C6 | none (random) | n/a | MIXED: est/se NaN, ci_lower finite, ci_upper NaN | UB: confidence NaN index cast | DROP / param THROW | FIX-UB |
| `permutation_test_correlation` | C6 | none (random) | n/a | MIXED: observed NaN, p_value 0.002 (drop 0.317) | n/a | DROP (pairwise) | FIX |
| `permutation_test_paired` | C6 | none (random) | n/a | MIXED: observed NaN, p_value 0.002 (drop 0.954) | n/a | DROP (pairwise) | FIX |
| `permutation_test_two_sample` | C6 | none (random) | n/a | MIXED: observed NaN, p_value 0.002 (spuriously significant; drop gives 0.954) | n/a (n_permutations is size_t) | DROP | FIX |

## robust.hpp

10 functions (OK 3, FIX 0, FIX-UB 7, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `biweight_midvariance` | C1 | closed form (median, mad, c=9) | NA | POSDEP mid:VALUE(0) first:VALUE(0.239) + UB | c=NaN: VALUE 0 (all weights dropped) | NaN; c NaN THROW (#6) | FIX-UB |
| `cooks_distance` | C2 | closed form res^2*h/(p*mse*(1-h)^2) (cooks.distance(lm)) | ELEM (res and hat) | ELEM (residuals and hat_values) | mse=NaN: NaN (all); p is size_t | ELEM; mse NaN: NaN | OK |
| `detect_outliers_iqr` | C1 | closed form via quantile(x, c(.25,.75), type=7) | ERROR (quantile) | POSDEP mid:VALUE(fences collapse, all 15 points flagged) first:VALUE(shifted fences) + UB | k=NaN: NaN fences, 0 outliers | THROW (outlier flags, section 5); k NaN THROW | FIX-UB |
| `detect_outliers_modified_zscore` | C1 | closed form via median/mad | NA (fences, count) | POSDEP mid:NaN fences first:VALUE(shifted) + UB | threshold=NaN: NaN fences, 0 outliers | THROW (outlier flags, section 5); threshold NaN THROW | FIX-UB |
| `detect_outliers_zscore` | C1 | closed form (x-mean)/sd | NA (fences, count) | NaN fences, 0 outliers (mid and first) | threshold=NaN: NaN fences, 0 outliers | THROW (outlier flags, section 5); threshold NaN THROW (aligned in v0.5.0) | OK |
| `dffits` | C2 | closed form res*sqrt(h)/(sqrt(mse)*(1-h)) (dffits(lm)) | ELEM (res and hat) | ELEM (residuals and hat_values) | mse=NaN: NaN (all) | ELEM; mse NaN: NaN | OK |
| `hodges_lehmann` | C6 | wilcox.test(x, conf.int=TRUE)$estimate | DROP | POSDEP VALUE(3.05)/VALUE(2.925) + UB | - | DROP | FIX-UB |
| `mad` | C1 | mad(x, constant=1) | NA | POSDEP mid:NaN first:VALUE(0.35) + UB | - | NaN | FIX-UB |
| `mad_scaled` | C1 | mad(x) | NA | POSDEP mid:NaN first:VALUE(0.519) + UB | - | NaN | FIX-UB |
| `winsorize` | C2 | pmin(pmax(x, lo), hi), lo/hi=quantile(sort(x)) (reference script) | ELEM (sort() drops NA before quantile) | POSDEP MIXED (NaN kept, wrong clip limits) + UB | limits=NaN: UB (float->size_t cast via percentile); normal build VALUE (no clipping) | data ELEM; limits NaN THROW (#6) | FIX-UB |

## shape_of_distribution.hpp

6 functions (OK 6, FIX 0, FIX-UB 0, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `kurtosis` | C1 | e1071::kurtosis(x, type=2) | NA | NaN | precomputed_mean=NaN: NaN | NaN | OK |
| `population_kurtosis` | C1 | e1071::kurtosis(x, type=1) | NA | NaN | precomputed_mean=NaN: NaN | NaN | OK |
| `population_skewness` | C1 | e1071::skewness(x, type=1) | NA | NaN | precomputed_mean=NaN: NaN | NaN | OK |
| `sample_kurtosis` | C1 | e1071::kurtosis(x, type=2) | NA | NaN | precomputed_mean=NaN: NaN | NaN | OK |
| `sample_skewness` | C1 | e1071::skewness(x, type=2) | NA | NaN | precomputed_mean=NaN: NaN | NaN | OK |
| `skewness` | C1 | e1071::skewness(x, type=2) | NA | NaN | precomputed_mean=NaN: NaN | NaN | OK |

## special_functions.hpp

16 functions (OK 16, FIX 0, FIX-UB 0, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `beta` | C7 | beta(a,b) | n/a | n/a | NaN | NaN | OK |
| `betainc` | C7 | pbeta(x,a,b) | NA | NaN | NaN | NaN | OK |
| `betainc_impl` | C7 | pbeta(x,a,b) | NA | NaN | NaN | NaN | OK |
| `betaincinv` | C7 | qbeta(p,a,b) | NA | NaN | NaN | NaN | OK |
| `erf` | C7 | 2*pnorm(x*sqrt(2))-1 | NA | NaN | n/a | NaN | OK |
| `erfc` | C7 | 2*pnorm(-x*sqrt(2)) | NA | NaN | n/a | NaN | OK |
| `gammainc_lower` | C7 | pgamma(x,a) | NA | NaN | NaN | NaN | OK |
| `gammainc_lower_inv` | C7 | qgamma(p,a) | NA | NaN | NaN | NaN | OK |
| `gammainc_upper` | C7 | pgamma(x,a,lower.tail=FALSE) | NA | NaN | NaN | NaN | OK |
| `lbeta` | C7 | lbeta(a,b) | n/a | n/a | NaN | NaN | OK |
| `lgamma` | C7 | lgamma(x) | NA | NaN | n/a | NaN | OK |
| `lgamma_impl` | C7 | lgamma(x) | NA | NaN | n/a | NaN | OK |
| `norm_cdf` | C7 | pnorm(x) | NA | NaN | n/a | NaN | OK |
| `norm_quantile` | C7 | qnorm(p) | NA | NaN | n/a | NaN | OK |
| `norm_sf` | C7 | pnorm(x,lower.tail=FALSE) | NA | NaN | n/a | NaN | OK |
| `tgamma` | C7 | gamma(x) | NA | NaN | n/a | NaN | OK |

## survival.hpp

4 functions (OK 2, FIX 0, FIX-UB 0, FIX-HANG 2).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `kaplan_meier` | C6 | survfit(Surv(t,e)~1,conf.type="plain") | DROP | HANG | n/a | DROP | FIX-HANG |
| `logrank_test` | C6 | survdiff(Surv(t,e)~g) | DROP | DROP | n/a | DROP | OK |
| `median_survival_time` | C1 | summary(survfit)$table["median"] | DROP (upstream survfit) | HANG via kaplan_meier input; direct struct: NaN survival skipped, NaN time returned | n/a | follows kaplan_meier (DROP) | OK |
| `nelson_aalen` | C6 | cumsum(d/n) via survfit | DROP | HANG | n/a | DROP | FIX-HANG |

## time_series.hpp

12 functions (OK 8, FIX 4, FIX-UB 0, FIX-HANG 0).

| Function | Class | R counterpart | R (NA data) | v0.4.0 (NaN data) | v0.4.0 (NaN parameter) | Policy (v0.5.0) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `acf` | C2 | acf(x,plot=FALSE) | ERROR:missing values in object (na.fail) | MIXED: [1,NaN,NaN,...] | n/a | THROW (R na.fail) (#17) | FIX |
| `autocorrelation` | C1 | acf(x,plot=FALSE)$acf[lag+1] | ERROR:missing values in object (na.fail) | NaN (lag 0 returns 1) | n/a | THROW (R na.fail) (#17) | FIX |
| `diff` | C2 | diff(x,differences=order) | ELEM | ELEM | n/a | ELEM | OK |
| `exponential_moving_average` | C2 | closed form recursion | CUM | CUM | MIXED: first element finite, rest NaN | CUM | OK |
| `lag` | C2 | closed form x[1:(n-k)] | ELEM | ELEM | n/a | ELEM | OK |
| `mae` | C1 | mean(abs(a-p)) | NA | NaN | n/a | NaN | OK |
| `mape` | C1 | mean(abs((a-p)/a))*100 | NA | NaN | n/a | NaN | OK |
| `moving_average` | C2 | stats::filter(x,rep(1/w,w),sides=1) | ELEM (only windows containing NA) | CUM (running-sum update: NaN poisons all later windows; first-position NaN makes all NaN) | n/a | ELEM | FIX |
| `mse` | C1 | mean((a-p)^2) | NA | NaN | n/a | NaN | OK |
| `pacf` | C2 | pacf(x,plot=FALSE) | ERROR:missing values in object (na.fail) | MIXED: [1,NaN,NaN,...] | n/a | THROW (R na.fail) (#17) | FIX |
| `rmse` | C1 | sqrt(mean((a-p)^2)) | NA | NaN | n/a | NaN | OK |
| `seasonal_diff` | C2 | diff(x,lag=period) | ELEM | ELEM | n/a | ELEM | OK |
