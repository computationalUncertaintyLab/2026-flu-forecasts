
#--MODEL

#mcandrew
"""
Clustered functional GP: one template per cluster + novel cluster.

Historical seasons in cluster k, with the leading eigenfunction only
and scale fixed at 1 (μ_k is already the cluster mean):

    y_s(τ) = [ μ_k(τ) + E_k(τ) c_k ] + u_s(τ)

Current season is a baseline plus a convex combination of the templates'
excess over their own floors, each with its own amplitude shrunk toward 1:

    μ_c(t) = b + Σ_k w_k a_k ([μ_k + E_k c_k](τ(t)) − b_k),   w ~ Dirichlet(π)
    a_k ~ LogNormal(0, 0.5)

b_k is template k's own off-season floor and b is the target location's, both
fixed low quantiles of the data. Without that split a_k multiplies the floor too,
so pushing a_k up to recover a flattened peak drags the off-season far negative.

Counts stay on the natural scale (no log1p).

v3 of cluster_stat. Same model, but the residual scale is a penalized B-spline
(P-spline) instead of linear interpolation between 5 fixed knots:

    log σ(τ) = B(τ) β_hat + (B(τ) L) z,    z ~ N(0, I)

B is a cubic B-spline on *equally spaced* knots, roughly one every
SIGMA_WEEKS_PER_KNOT weeks, so σ can change on that timescale rather than only
at the 5 places the old knots happened to sit. Equal spacing is what makes this
a P-spline: a difference penalty on adjacent coefficients only means "smooth"
when the coefficients it links are a constant distance apart.

The penalty is the usual squared second difference, which as a prior is
Gaussian with precision λ P'P. That is factored once up front -- L is its
Cholesky -- so sampling is iid standard normals and the model costs a single
matmul. Writing the same prior as a random walk with its own sampled scale
(Δ²β ~ N(0, s_pen)) is mathematically equivalent but hands NUTS a funnel, and
with a T x T Cholesky per historical season on every leapfrog step that is the
difference between seconds and many minutes.

λ is chosen by GCV. β_hat is the penalized fit of the empirical log SD, so the
curve starts from a real residual scale rather than free basis weights, and the
prior width around it combines the sampling error of that SD, 1/sqrt(2(n-1)) in
log units, with how much of the empirical curve the smooth fit leaves behind.
Every floor is relative to the sd of Y, since the runner hands this class
per-location standardized data.

Correlation is Matérn-3/2;
K(τ,τ') = σ(τ)σ(τ') k(τ,τ'). An RBF kernel on this grid is smooth
enough to extrapolate a seasonal residual into bands tens of times
the data. FPCA scores use the eigenvalue scale directly (no extra
sigma2_c).

Pinned copy with per-cluster a_c: forecast/puca_with_eigenfunctions.py
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAX_ENABLE_X64"] = "True"

import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from scipy.interpolate import BSpline

class cluster_stat3(object):

    #--Fewest seasons a cluster may hold. Matches the n < 4 fallback in
    #--_loo_score_shrinkage: below this the leave-one-out cannot be computed, so a
    #--cluster that small cannot produce a usable eigenvalue either.
    MIN_CLUSTER_SEASONS = 4

    #--KMeans was unseeded, so the clustering -- and therefore every template, the
    #--eigenfunctions and the number of amplitudes -- changed run to run.
    KMEANS_SEED = 20200320

    #--Pseudo-degrees-of-freedom given to the pooled peak-timing variance when
    #--shrinking a location's own estimate toward it. 3 is one location's worth of
    #--df at four seasons, so a four-season location gets a 50/50 blend and a
    #--longer record keeps more of its own.
    PEAK_SD_POOL_DF = 3.0

    #--Floor on the peak-week sd, as a fraction of the season. 0.067 * 44 = 2.9
    #--weeks, the lower quartile of the peak-week sd across all 52 locations in
    #--the target file (median 4.11, IQR 2.96-4.57).
    #--This is needed on top of the pooling because donors are chosen for
    #--correlation with the target, and peak timing is a seasonal phenomenon: in
    #--the Illinois pool every donor peaked in week 13 in the same season and week
    #--23 in another, so the donors are not independent replicates of timing
    #--variability and pooling over them adds little. Hawaii's pool happens never
    #--to have seen an early or late season, which is weak evidence that Hawaii
    #--cannot have one when week-13 peaks demonstrably occurred elsewhere.
    #--A national pool handed in by the runner would be the better fix; this class
    #--only ever receives D.
    PEAK_SD_FLOOR_FRAC = 0.067

    #--Fraction of the season a past column must cover to count as a season at
    #--all. 0.75 keeps the one column in the Montana file with 5 weeks missing
    #--(39/44) and drops the t0-week current seasons of the donor locations.
    MIN_SEASON_COVERAGE = 0.75

    #--Target spacing of the sigma spline knots, in weeks. The disease grid packs
    #--T weeks into tau in [0, 2], so this is converted rather than used directly.
    SIGMA_WEEKS_PER_KNOT = 5

    #--Sharpness of the smooth ceiling on log sigma. Larger is closer to a hard
    #--clamp. At 6 the cap is inert over the range sigma actually occupies (0.2%
    #--at half the ceiling, under 0.01% below that) and costs 11% at the ceiling
    #--itself, which is the price of keeping a gradient there.
    SIGMA_CAP_SHARPNESS = 6.0

    def __init__(self, D, interpolate=True, t0=None, CAP=None):
        D = self.drop_partial_past(D)
        T, S = D.shape
        self.T = T
        self.S = S
        self.D = D

        if CAP is None:
            self.CAP=S
        else:
            self.CAP = CAP

        self.disease_grid = np.linspace(0, 2, T)
        self.delta_tau = self.disease_grid[1]
        self.calendar_grid = np.arange(T)
        self.delta_t = self.calendar_grid[1]

        if interpolate:
            D = self.interp_Y(D)
            self.Dinterp = D

        Y = D
        target_y = Y[:, -1]

        if t0 is None:
            nan_idx = np.argwhere(np.isnan(np.asarray(target_y, dtype=float)))
            if nan_idx.size == 0:
                raise ValueError(
                    "Last season is fully observed (no NaNs), so there is no t0. "
                    "ENDDATE is at or after the season end, or off-season weeks were "
                    "dropped and left a complete last column."
                )
            t0 = int(nan_idx[0])
        self.t0 = t0
        self.Y = Y
        self.target_y = target_y

    def fit(self, mode=0, num_warmup=1500, num_samples=1500, predict=True):
        Yscaled = self.scale(self.Y)
        self.Yscaled = Yscaled

        self.past_peaks = self.compute_peaks(Yscaled)
        self.peak_mean, self.peak_std = self.compute_peak_prior(Yscaled)

        mean_t = self.compute_mean_func(Yscaled, self.disease_grid)
        self.mean_t = mean_t
        self.mu_c = mean_t - np.mean(mean_t)

        #--cluster on the residuals
        eps      = self.Ys_disease_time.T - mean_t[:, None]
        n_season = eps.shape[-1]
        if n_season == 1 or self.CAP == 1:
            groups = np.zeros(max(n_season, 1), dtype=int)
            nclust = 1
        else:
            #--A cluster is a forecast template: its mean is read directly as
            #--mu_k and its leading eigenvalue sets the prior on this season's
            #--shape score. Both are useless from two or three seasons. Searching
            #--k up to S_past - 1 let Hawaii split 19 seasons into 5 clusters, two
            #--of them with n = 2, which produced templates averaged from two
            #--curves and six amplitudes to estimate from four observations. Bound
            #--k so every cluster can hold at least MIN_CLUSTER_SEASONS, and reject
            #--any k that puts a cluster below it even so -- silhouette optimizes
            #--separation and is perfectly happy with a cluster of two.
            k_hi = int(min(self.CAP, max(n_season // self.MIN_CLUSTER_SEASONS, 1)))

            #--Seeded, and n_init raised off the default. The previous version
            #--scored one unseeded KMeans per k and then refit a *second* unseeded
            #--KMeans to get the labels, so the labels did not necessarily come
            #--from the fit whose silhouette selected k.
            best = None
            for k in range(2, k_hi + 1):
                lab = KMeans(
                    n_clusters   = k,
                    n_init       = 10,
                    random_state = self.KMEANS_SEED,
                ).fit(eps.T).labels_
                if np.bincount(lab, minlength=k).min() < self.MIN_CLUSTER_SEASONS:
                    continue
                score = float(np.mean(silhouette_samples(eps.T, lab)))
                if best is None or score > best[0]:
                    best = (score, k, lab)

            if best is None or best[0] < 0.25:
                groups = np.zeros(n_season, dtype=int)
                nclust = 1
            else:
                nclust = best[1]
                groups = best[2]
            self.silhouette = None if best is None else best[0]
        self.groups = np.asarray(groups, dtype=int)
        groups      = self.groups

        Ys_disease_time_clustered = {n: [] for n in range(nclust)}
        for y, z in zip(self.Ys_disease_time, groups):
            Ys_disease_time_clustered[z].append(y)
        Ys_disease_time_clustered = {
            k: np.array(x).T for k, x in Ys_disease_time_clustered.items()
        }
        self.Ys_disease_time_clustered = Ys_disease_time_clustered

        cluster_to_mean        = {}
        cluster_to_eigen_funcs = {}
        cluster_to_lambdas     = {}
        cluster_to_loo_shrink  = {}
        for z, y in Ys_disease_time_clustered.items():
            mean_k             = np.mean(y, axis=1)
            cluster_to_mean[z] = mean_k

            es, lambdas, eigenfuncs = self.compute_covariance_and_eigenfunction(
                y=y, mean_t=mean_k
            )
            K_k = 1
            #--Non-negativity only. The old floor of 1e-2 was an absolute constant
            #--against per-location standardized data (see scale()), where the
            #--eigenvalues are ~0.03 -- so it was a third of the signal, and it
            #--silently clamped away the leave-one-out shrinkage below. A lam of 0
            #--is not degenerate anyway: the rank-k term vanishes, K stays positive
            #--definite, and the current-season score becomes a harmless no-op.
            lam_1 = max(float(lambdas[0]), 0.0) if lambdas.size else 0.0

            #--Shrink toward the out-of-sample score magnitude, so the prior on the
            #--current season's shape is calibrated to a season that did not help
            #--build the eigenfunctions.
            shrink = self._loo_score_shrinkage(y, K_k)

            cluster_to_eigen_funcs[z] = eigenfuncs[:, :K_k]
            cluster_to_lambdas[z]     = np.array([lam_1]) * shrink
            cluster_to_loo_shrink[z]  = np.asarray(shrink, dtype=float)

        # K+1 novel cluster: no past seasons, pooled mean, no FPCA
        novel_k                            = nclust
        nclust_total                       = nclust + 1
        Ys_disease_time_clustered[novel_k] = np.zeros((self.T, 0))
        cluster_to_mean[novel_k]           = mean_t

        self.n_hist = nclust
        self.nclust = nclust_total
        self.cluster_to_mean = cluster_to_mean
        self.cluster_to_eigen_funcs = cluster_to_eigen_funcs
        self.cluster_to_lambdas     = cluster_to_lambdas
        self.cluster_to_loo_shrink  = cluster_to_loo_shrink

        #--Baseline / excess split. blend() multiplies each template by w_k a_k, so
        #--with no additive level `a` rescales the off-season floor along with the
        #--epidemic. Averaging aligned seasons flattens a template peak by ~1.35x,
        #--so `a` was driven to 2.7 to recover the observed peak, and that same 2.7
        #--turned a floor of -0.60 into -1.61 -- about -42 admissions. 69% of
        #--forecast draws came back negative and 27 of 40 weekly medians clipped to
        #--zero. Splitting each template into its own floor plus the excess above it
        #--lets `a` scale only the epidemic. The floors are fixed quantiles of the
        #--data rather than free parameters, so unlike a sampled intercept they
        #--cannot trade off against w and a.
        #--Cluster floors are the template MINIMUM, not a low quantile. A quantile
        #--leaves the template below its own floor at the bottom of the declining
        #--limb, and a_k ~ 2.6 amplifies that dip: with the 10th percentile the
        #--forecast still ran to -26 and -48 admissions over weeks 14-24. Using the
        #--minimum makes the excess non-negative everywhere, so mu >= base for every
        #--tau and every a_k >= 0 and the mean can no longer go below baseline.
        self.cluster_base = {
            k: float(np.min(np.asarray(v, dtype=float)))
            for k, v in cluster_to_mean.items()
        }
        BASE_Q = 0.10

        #--Target's own off-season level, taken from its past seasons on this same
        #--scale, so the forecast reverts to this location's baseline and not to the
        #--donors' pooled one.
        idx   = self.target_location_past_idx()
        tcols = np.asarray(Yscaled[:, :-1], dtype=float)
        tcols = tcols[:, idx] if idx.size else tcols
        self.target_base = float(np.nanquantile(tcols, BASE_Q))

        self._init_sigma_spline(
            Ys_disease_time_clustered,
            cluster_to_mean,
            eigen  = cluster_to_eigen_funcs,
            shrink = cluster_to_loo_shrink,
        )

        base_key = jax.random.PRNGKey(20200320)
        key = jax.random.fold_in(base_key, 1)

        #--Two near-independent groups, so a dense mass over the union just pays
        #--for covariances that are ~0. Within the mean group w_k and a_k enter
        #--blend() only as the product w_k*a_k, peak shifts the tau the templates
        #--are read at, and z_c deforms those templates, so they are strongly
        #--correlated. sigma_knot is linearly interpolated (neighbouring knots
        #--correlate) and the nugget fraction trades off against its overall level.
        mean_sites  = ("peak", "w", "a", "delta") + tuple(
            "z_c_{:d}".format(k) for k in range(self.n_hist)
        )
        scale_sites = ("sigma_z", "nugget_logit")

        nuts_kernel = NUTS(
            self.model,
            init_strategy  = init_to_median(num_samples=100),
            dense_mass     = [mean_sites, scale_sites],
            #--Depth 12 allows 4095 steps. Post-adaptation the sampler only asks
            #--for 15-47, so that ceiling is reachable only during the
            #--pre-adaptation warmup where the step size is still ~1e-3; dropping
            #--to 8 (255 steps) measured 2.5x faster with an unchanged posterior.
            max_tree_depth = 12,
        )
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=1,
            chain_method="parallel",
        )
        mcmc_kwargs = dict(
            y_obs       = Yscaled[: self.t0, -1],
            y_past      = {k: jnp.asarray(v) for k, v in Ys_disease_time_clustered.items()},
            y_past_all  = jnp.asarray(self.Ys_disease_time.T),
            mean_k      = {k: jnp.asarray(v) for k, v in cluster_to_mean.items()},
            E           = {k: jnp.asarray(v) for k, v in cluster_to_eigen_funcs.items()},
            lambdas     = {k: jnp.asarray(v) for k, v in cluster_to_lambdas.items()},
            times       = jnp.asarray(self.calendar_grid),
            grid        = jnp.asarray(self.disease_grid),
            t0=int(self.t0),
            peak_mean=float(self.peak_mean),
            peak_std=float(self.peak_std),
        )
        mcmc.run(
            key,
            extra_fields=("potential_energy", "energy", "num_steps"),
            forecast=False,
            **mcmc_kwargs,
        )
        self.mcmc = mcmc
        self.mcmc_kwargs = mcmc_kwargs
        self.extra_fields = {
            k: np.asarray(v) for k, v in mcmc.get_extra_fields().items()
        }
        self.posterior_samples = mcmc.get_samples()
        mcmc.print_summary()

        if not predict:
            self.forecast = None
            return

        pred = Predictive(self.model, posterior_samples=self.posterior_samples)(
            jax.random.fold_in(key, 1), forecast=True, **mcmc_kwargs
        )
        samples_bayes = np.asarray(pred["y_pred"])
        y = samples_bayes * self.global_std + self.global_mean
        self.forecast = y#np.clip(y, 0, None)

    def model(
        self,
        y_obs      = None,
        y_past     = None,
        y_past_all = None,
        mean_k     = None,
        E          = None,
        lambdas    = None,
        times=None,
        grid=None,
        t0=None,
        peak_mean=None,
        peak_std=None,
        forecast=False,
    ):
        T, Spast = y_past_all.shape
        nclust = len(y_past)
        n_hist = nclust - 1
        #--Fitted from the residual autocorrelation in _fit_length_scale, not the
        #--hardcoded 1/pi that overstated adjacent-week correlation and forced
        #--sigma up to compensate.
        ell    = getattr(self, "sigma_ell", float(self.ELL_DEFAULT))

        # One calendar peak for the current season. Clusters are shape-only.
        peak_c = numpyro.sample(
            "peak",
            dist.TruncatedNormal(peak_mean, peak_std, low=0, high=T - 1),
        )
        tau_c        = 1 + (times - peak_c) / T
        tau_observed = tau_c[:t0]
        h_fut        = tau_c if t0 == 0 else tau_c[t0:]

        hist_counts = jnp.array(
            [y_past[k].shape[1] for k in range(n_hist)], dtype=jnp.float32
        )
        # Concentration is the season count itself, not a normalized share.
        # Dividing by the sum pins the total at 1, the most diffuse Dirichlet
        # there is, and with every component then below 1 the density is
        # corner-seeking -- w came back indistinguishable from its prior.
        alpha = jnp.concatenate([hist_counts, jnp.array([1.0])])
        w     = numpyro.sample("w", dist.Dirichlet(alpha))

        # P-spline on log σ, in whitened form. The second-difference penalty is
        # a Gaussian prior on the coefficients, so it gets factored once in
        # _init_sigma_spline and sampling reduces to iid standard normals:
        #
        #     log σ(τ) = B β_hat + (B L) z,    z ~ N(0, I)
        #
        # Sampling the penalty as a random walk with its own scale (s_pen * z) is
        # the same prior but gives NUTS a funnel, which collapses the step size
        # and drives the tree to max depth. Every leapfrog step Choleskys a T x T
        # K_hist per historical season, so that costs whole minutes. Here the
        # posterior geometry is unit-scale and there is one matmul per step.
        z_sig = numpyro.sample(
            "sigma_z",
            dist.Normal(0.0, jnp.ones(self.sigma_n_basis)).to_event(1),
        )

        log_sigma_grid = (
            jnp.asarray(self.sigma_logsd_grid) + jnp.asarray(self.sigma_M) @ z_sig
        )
        # Smooth cap rather than jnp.minimum. A hard clamp zeroes the gradient on
        # one side, and it was binding across a third of the season, so the sampler
        # was pushing sigma_z to 6-7 against a wall it got no gradient information
        # from. softplus saturates the same way but stays differentiable:
        # well below the ceiling it is the identity, well above it returns log_max.
        # SIGMA_CAP_SHARPNESS trades how abrupt the saturation is against how much
        # the cap pulls sigma down when it is nowhere near binding.
        log_max        = jnp.log(self.sigma_max)
        k              = self.SIGMA_CAP_SHARPNESS
        log_sigma_grid = log_max - jnp.logaddexp(0.0, k * (log_max - log_sigma_grid)) / k
        sigma_grid     = numpyro.deterministic("sigma_t", jnp.exp(log_sigma_grid))
        
        # Nugget as a FRACTION of sigma(tau)^2 rather than an absolute constant.
        # The week-to-week reporting noise is proportional to the local scale: it
        # correlates 0.917 with sigma(tau) across the within-cluster residuals and
        # spans the same 31x range. One additive sigma_obs had to serve both the
        # peak, where that noise is ~0.50, and the off-season, where it is ~0.008.
        # Off-season weeks outnumber peak weeks, so the likelihood drove sigma_obs
        # to 0 -- cheaper to drop the nugget than to be 8x too wide for most of the
        # season -- which left the peak with no nugget at all and made sigma(tau)
        # inflate to absorb the jaggedness instead. Sampled on the logit scale so
        # it stays in (0, 1) without a boundary, centred on the fraction fitted
        # jointly with ell from the residual autocorrelation.
        nug_logit = numpyro.sample(
            "nugget_logit",
            dist.Normal(self.sigma_nugget_logit, self.NUGGET_LOGIT_SD),
        )
        eta       = numpyro.deterministic("nugget_frac", 1.0 / (1.0 + jnp.exp(-nug_logit)))
        signal    = 1.0 - eta

        def sigma_at(tau):
            return jnp.interp(tau, grid, sigma_grid)

        # y = f + e with Var(f) = (1-eta) sigma^2 and Var(e) = eta sigma^2, so the
        # total marginal variance is sigma^2 however eta moves and the smooth part
        # carries (1-eta) of it. eta is white, so it is on the diagonal of K_OO and
        # K_FF but absent from the cross-covariance K_FO below.
        def K_het(x1, x2, s1, s2, diag=True):
            K = signal * self.scaled_rbf(x1, x2, s1, s2, length_scale=ell)
            if not diag:
                return K
            n = x1.shape[0]
            # Relative jitter. An absolute 1e-5 is ~1e-12 of a case-scale
            # variance and the conditional mean runs away.
            rel = 1e-4 * jnp.mean(s1 ** 2)
            K = K + jnp.diag(eta * s1 ** 2 + rel * jnp.ones(n))
            return 0.5 * (K + K.T)

        K_hist = K_het(grid, grid, sigma_grid, sigma_grid)

        # Historical seasons, with their per-season FPCA scores marginalized out.
        #
        # Karhunen-Loeve gives every season its own loading,
        #     y_s = mu_k + Phi_k c_s,   c_s ~ N(0, diag(lam_k))  independently per s
        # and the earlier code instead sampled ONE score per cluster and shared it
        # across all n_k members. The likelihood-optimal value of a shared score is
        # the mean of the per-season scores, and that mean is zero by construction,
        # because the scores are projections of deviations from mu_k: measured on
        # Montana the per-season scores had sd sqrt(lam_1) = 0.179 while their mean
        # was 1e-17. The shared score duly came back at 0.00 +/- 0.10 against a unit
        # prior, with 99.5% of its posterior precision supplied by this likelihood
        # insisting it stay at zero and 0.5% by the current season. It deformed the
        # template by under 3% of its peak where the data supports 28%.
        #
        # c_s is Gaussian and enters linearly, so integrating it out is exact, adds
        # no parameters, and turns the shape variation into a rank-K_k term on the
        # covariance. That is where it belonged: previously the only place this
        # variation could go was sigma(tau).
        for k in range(n_hist):
            Y_k   = y_past[k]
            n_k   = Y_k.shape[1]
            mu_k  = mean_k[k]
            E_k   = E[k]
            # Non-negativity only. A 1e-2 floor here would clamp away the
            # leave-one-out shrinkage applied in fit(), which took one cluster's
            # lam from 0.025 to 0.006.
            lam_k = jnp.maximum(jnp.asarray(lambdas[k]), 0.0)

            # (E_k * lam_k) @ E_k.T is sum_j lam_j phi_j phi_j', the covariance of
            # Phi_k c_s. One Cholesky per cluster rather than per season: the mean
            # and covariance are both constant within a cluster, so the n_k members
            # batch against a single factorization.
            # delta_sd^2 11' is the exact marginalization of a per-season baseline
            # offset delta_s ~ N(0, delta_sd^2), the same device as the scores above
            # and for the same reason: the current season gets a sampled `delta`, so
            # the historical seasons must be allowed the same freedom or sigma(tau)
            # would have to absorb their level variation instead.
            K_k = K_hist + (E_k * lam_k) @ E_k.T + self.delta_sd ** 2
            numpyro.sample(
                f"ll_hist_{k}",
                dist.MultivariateNormal(jnp.broadcast_to(mu_k, (n_k, T)), K_k),
                obs=Y_k.T,
            )

        # Current-season shape scores. Deliberately separate from the historical
        # seasons: there a score is a perturbation of a shared mean that mu_k
        # already optimizes, whereas here it is *this* season's own loading on the
        # cluster's modes of variation, and the only thing informing it is y_obs.
        # The prior scale sqrt(lam_k) is the between-season spread FPCA measured, so
        # one prior sd means "as far from the cluster mean as a typical season".
        tmpl_k = []
        for k in range(n_hist):
            E_k   = E[k]
            lam_k = jnp.maximum(jnp.asarray(lambdas[k]), 0.0)
            z_c_k = numpyro.sample(
                f"z_c_{k}",
                dist.Normal(0.0, jnp.ones(E_k.shape[1])).to_event(1),
            )
            c_k = numpyro.deterministic(f"c_{k}", z_c_k * jnp.sqrt(lam_k))
            tmpl_k.append(mean_k[k] + E_k @ c_k)
            numpyro.deterministic(f"mu_{k}", tmpl_k[k])

        tmpl_k.append(mean_k[n_hist])

        # One current-season amplitude per cluster, shrunk to 1; data can pull it away.
        a = numpyro.sample("a", dist.LogNormal(0.0, 1./2).expand([nclust]))

        # Baseline plus epidemic excess, so a_k scales only the epidemic and leaves
        # the off-season level alone. Reduces to the old pure product when a_k = 1
        # and the cluster floor equals the target's, so the historical likelihood
        # above (which fits tmpl_k directly) stays consistent with it.
        # Baseline offset for *this* season. target_base is the 10th percentile of
        # this location's past seasons, which is a reasonable centre but was being
        # used as a known constant. Because every E_k >= 0 the mean is built upward
        # from it, so the floor fixes the ratio the observed weeks must rise by --
        # see _level_prior_sd for why that is hypersensitive when the first
        # observation sits a couple of admissions above the floor. delta_sd is the
        # between-season spread of the off-season level, so one prior sd means
        # "this season's baseline is as far from typical as a typical season's is".
        delta  = numpyro.sample("delta", dist.Normal(0.0, self.delta_sd))
        base   = numpyro.deterministic("base", self.target_base + delta)
        base_k = self.cluster_base

        def blend(tau):
            mu = jnp.full(tau.shape[0], base)
            for k in range(nclust):
                excess = jnp.interp(tau, grid, tmpl_k[k]) - base_k[k]
                mu     = mu + w[k] * a[k] * excess
            return mu

        if t0 > 0:
            mu_s = blend(tau_observed)
            numpyro.deterministic("mu_this_season", mu_s)
            s_obs = sigma_at(tau_observed)
            K_OO = K_het(tau_observed, tau_observed, s_obs, s_obs)
            numpyro.sample(
                "this_season",
                dist.MultivariateNormal(mu_s, K_OO),
                obs=y_obs,
            )

        if forecast:
            mu_fut = blend(h_fut)
            n_fut = h_fut.shape[0]
            s_fut = sigma_at(h_fut)
            KFF = K_het(h_fut, h_fut, s_fut, s_fut)
            if t0 == 0:
                numpyro.sample(
                    "y_pred", dist.MultivariateNormal(mu_fut, KFF)
                )
            else:
                residual = y_obs - mu_s
                # Signal only, no diagonal: the nugget is white reporting noise, so
                # it carries nothing from the observed weeks to the future ones.
                K_FO = K_het(h_fut, tau_observed, s_fut, s_obs, diag=False)
                mu_cond = mu_fut + K_FO @ jnp.linalg.solve(K_OO, residual)
                K_cond = KFF - K_FO @ jnp.linalg.solve(K_OO, K_FO.T)
                K_cond = 0.5 * (K_cond + K_cond.T)
                rel = 1e-4 * jnp.mean(s_fut ** 2)
                K_cond = K_cond + rel * jnp.eye(n_fut)
                numpyro.sample(
                    "y_pred", dist.MultivariateNormal(mu_cond, K_cond)
                )

    def matern32_kernel_j(self, x1, x2, length_scale=1.5, variance=1.0):
        """Matérn-3/2. Exponential tails, so the forecast reverts to the template.

        An RBF conditional mean on this disease-time grid rings outside
        the observed weeks and the 95% band leaves the data.
        """
        r = jnp.abs(x1[:, None] - x2[None, :])
        a = jnp.sqrt(3.0) * r / length_scale
        return variance * (1.0 + a) * jnp.exp(-a)

    def scaled_rbf(self, x1, x2, s1, s2, length_scale=1.0):
        corr = self.matern32_kernel_j(x1, x2, length_scale=length_scale, variance=1.0)
        return s1[:, None] * corr * s2[None, :]

    #--Most a leave-one-out score ratio is allowed to shrink the FPCA scale, as an
    #--sd ratio. Also the value used when a cluster has too few seasons to run the
    #--leave-one-out at all, since an eigenvalue from three curves is worth less
    #--than the most overfit case we have actually measured.
    #--Fallback prior sd for the current season's baseline offset, as a fraction of
    #--the mean pointwise residual sd, used only when the between-season level
    #--variation cannot be estimated (no cluster holds two seasons).
    DELTA_SD_FALLBACK_FRAC = 0.25

    LOO_SHRINK_FLOOR = 0.20

    #--Fallback length scale, in units of tau, if the residuals are too thin to
    #--estimate one. This was the hardcoded value for every location.
    ELL_DEFAULT = 1.0 / np.pi

    #--Fallback nugget fraction, used on the same thin-residual path as
    #--ELL_DEFAULT, and prior width for it on the logit scale. 0.75 around a
    #--fitted 0.156 spans roughly 0.04 to 0.45, wide enough that the data can
    #--overrule the ACF estimate but not so wide that 0 and 1 are in reach.
    NUGGET_FRAC_DEFAULT = 0.15
    NUGGET_LOGIT_SD     = 0.75

    def _loo_score_shrinkage(self, Y, n_comp):
        """Leave-one-out correction to the FPCA eigenvalues.

        phi_1 is the direction of maximum variance *in this sample*, so a season
        that was not used to build it loads less on it than the in-sample
        eigenvalue advertises. lam_k is nevertheless used as the prior variance of
        the current season's score, and the current season is by definition one
        that was not used -- so that prior is systematically too wide, and the
        error scales with 1/n the way eigenvalue overfitting does. Measured here,
        the ratio of out-of-sample RMS score to in-sample sqrt(lam_1) was 0.50 for
        a 4-season cluster and 0.85 for a 15-season one: a prior twice as wide as
        a genuinely new season warrants.

        This has to be measured out of sample. Adding a hierarchical scale on
        lam_k would not find it, because such a scale is fitted against the same
        seasons that defined phi_1, and phi_1 was chosen to maximize their
        variance along it -- the historical likelihood reports a factor of 1 by
        construction.

        Returns a per-component factor to multiply lam by, i.e. a squared sd
        ratio, never above 1: the in-sample eigenvalue is an upper bound on the
        predictive variance along the in-sample direction.
        """
        Y = np.asarray(Y, dtype=float)
        T = self.T
        n = Y.shape[1]
        wq = self.delta_tau * np.array([0.5] + [1] * (T - 2) + [0.5])

        _, lam_in, _ = self.compute_covariance_and_eigenfunction(
            y=Y, mean_t=Y.mean(axis=1)
        )
        n_comp = int(min(n_comp, lam_in.size))
        if n_comp < 1:
            return np.ones(0)
        sd_in = np.sqrt(np.maximum(lam_in[:n_comp], 1e-18))

        #--Under four seasons leaves too few held-out points for an RMS to mean
        #--anything, so take the most shrunk case rather than trust the eigenvalue.
        if n < 4:
            return np.full(n_comp, self.LOO_SHRINK_FLOOR ** 2)

        acc = np.zeros(n_comp)
        for s in range(n):
            Yo   = np.delete(Y, s, axis=1)
            mu_o = Yo.mean(axis=1)
            _, _, ef_o = self.compute_covariance_and_eigenfunction(y=Yo, mean_t=mu_o)
            #--Residual of the held-out season about the held-in mean, formed the
            #--same way compute_covariance_and_eigenfunction forms its own.
            X    = np.column_stack([np.ones(T), mu_o])
            coef, _, _, _ = np.linalg.lstsq(X, Y[:, s], rcond=None)
            e    = Y[:, s] - X @ coef
            #--Squared, so the arbitrary sign of each held-in eigenfunction drops
            #--out and the scores can be pooled across folds.
            acc += (ef_o[:, :n_comp].T @ (wq * e)) ** 2
        rms = np.sqrt(acc / n)

        ratio = np.clip(rms / sd_in, self.LOO_SHRINK_FLOOR, 1.0)
        return ratio ** 2

    def _fit_length_scale(self, R):
        """Matern-3/2 length scale from the within-cluster residual autocorrelation.

        The hardcoded 1/pi is 6.8 weeks on a 44-week grid, which puts the
        correlation between adjacent weeks at 0.973 when the residuals actually
        sit near 0.82. That gap is not cosmetic: K^-1 divides adjacent-week
        differences by roughly (1 - rho), so overstating rho by that much makes
        genuinely jagged residuals look impossible, and the only free parameter
        able to rescue the density is sigma. On Montana it bought a sigma whose
        level broke its own prior by 4.4 sd, and the same near-singular K^-1 is
        what made `peak` come back at +/- 0.02 weeks from four noisy points.

        R is (T, n_res): residuals about the cluster mean, disease time down the
        rows and seasons across the columns.

        Also sets sigma_nugget_frac, the white-noise share of the variance, which
        is fitted here because leaving it out biases ell: see the comment on the
        grid search below.
        """
        self.sigma_nugget_frac = float(self.NUGGET_FRAC_DEFAULT)
        self.sigma_ell_acf     = None
        if R is None or R.ndim != 2 or R.shape[1] < 3:
            return float(self.ELL_DEFAULT)

        grid = np.asarray(self.disease_grid, dtype=float)
        dtau = float(grid[1] - grid[0])
        T    = R.shape[0]

        #--Standardize each tau so the heteroskedasticity, which sigma(tau) already
        #--models, does not let the epidemic peak dominate the correlation estimate.
        sd = np.nanstd(R, axis=1, keepdims=True)
        Rs = R / np.maximum(sd, 1e-12)

        lags  = np.arange(1, max(T // 3, 2))
        acf   = np.array([np.nanmean(Rs[:-L] * Rs[L:]) for L in lags])
        wt    = (T - lags).astype(float)          # pair counts
        ok    = np.isfinite(acf)
        if ok.sum() < 2:
            return float(self.ELL_DEFAULT)
        lags, acf, wt = lags[ok], acf[ok], wt[ok]

        #--Fit a nugget fraction alongside ell. The empirical ACF is 1 at lag 0 and
        #--0.82 at lag 1, and a Matern-3/2 is continuous, so it cannot drop that
        #--fast. Fitting ell alone makes it absorb the step by shortening, which
        #--mis-states the correlation the GP actually needs between weeks. The step
        #--is the white reporting noise the nugget is there to model, so let it take
        #--the step and leave ell to describe the smooth part.
        #--For a given ell the amplitude is a linear least squares problem, so the
        #--grid search stays one-dimensional.
        r     = lags * dtau
        best  = None
        for ell in np.logspace(np.log10(dtau), np.log10(2.0), 200):
            a   = np.sqrt(3.0) * r / ell
            k   = (1.0 + a) * np.exp(-a)
            den = float(np.sum(wt * k * k))
            amp = float(np.sum(wt * acf * k)) / den if den > 0 else 1.0
            amp = float(np.clip(amp, 0.3, 1.0))
            sse = float(np.sum(wt * (acf - amp * k) ** 2))
            if best is None or sse < best[0]:
                best = (sse, float(ell), amp)

        ell                     = best[1]
        self.sigma_nugget_frac  = float(1.0 - best[2])
        #--Keep it at least a grid step, or the kernel is a nugget and the GP
        #--conditional mean carries no information between weeks.
        ell = float(np.clip(ell, 1.5 * dtau, 2.0))
        self.sigma_ell_acf      = (lags * dtau, acf)
        self.sigma_ell_weeks    = ell / dtau
        return ell

    def _level_prior_sd(self, offsets, emp_sd):
        """Prior sd for the current season's baseline offset.

        `target_base` is a fixed 10th percentile of this location's past seasons
        with no uncertainty attached, and blend() builds the mean upward from it as
        base + sum_k w_k a_k E_k with every E_k >= 0. So the model can match the
        *level* of the observed weeks freely but not their *ratio* above the floor:
        with u_j = y_j - base, the fitted excess must satisfy

            u_last / u_first  in  [min_k R_k, max_k R_k],
            R_k(tau) = E_k(tau + dtau) / E_k(tau),

        because a positive combination of the E_k is bounded by their extremes.
        That requirement is violently sensitive to the floor when the first
        observation sits just above it: d(ratio)/d(base) = (y_last - y_first) /
        u_first^2, which for Hawaii in 2026/2027 is 656 per unit, or 27 per
        admission, because u_first is 1.5 admissions. The required ratio came out
        at 41.7 against a best available 34.9 -- infeasible, which the sampler
        could only resolve by sliding `peak` to week 6 and fitting badly. A shift
        of 0.3 admissions in the floor closes that gap.

        So let the floor move, with a scale taken from how much the off-season
        level actually varies between seasons: project each season's residual onto
        the constant function in L2(w) and take the sd of those projections,
        inflated by (1 + 1/n) because this is a prior for a season that has not
        happened yet rather than a description of the ones that have.
        """
        if offsets:
            d = np.concatenate([np.asarray(o, dtype=float).ravel() for o in offsets])
            d = d[np.isfinite(d)]
            if d.size > 1:
                s = float(np.std(d, ddof=1)) * np.sqrt(1.0 + 1.0 / d.size)
                if np.isfinite(s) and s > 0.0:
                    return s
        return float(self.DELTA_SD_FALLBACK_FRAC * np.nanmean(emp_sd))

    def _init_sigma_spline(self, clustered, means, eigen=None, shrink=None):
        #--Residuals about the cluster mean, with the eigenfunction component
        #--*partially* projected out. The historical likelihood carries the FPCA
        #--variation explicitly as a rank-K_k term on the covariance, so sigma(tau)
        #--must describe only what is left over -- fitting it to the raw residuals
        #--would count the same between-season shape variation twice and the prior
        #--on sigma would then fight the rank-K_k term.
        #--Partially, because the leave-one-out shrinkage means the rank-K_k term
        #--only claims a fraction f of the variance along phi_j. Scaling that
        #--component by sqrt(1 - f) leaves lam_j (1 - f) of it for sigma: f = 1
        #--removes it entirely, f = 0 leaves it untouched.
        #--The projection uses the L2(w) inner product the eigenfunctions are
        #--orthonormal in, i.e. the trapezoid weight, not the Euclidean one.
        wq = self.delta_tau * np.array([0.5] + [1] * (self.T - 2) + [0.5])
        wn = wq / wq.sum()
        pieces, offsets = [], []
        for z, y in clustered.items():
            if y.shape[1] == 0:
                continue
            res = np.asarray(y, float) - np.asarray(means[z], float)[:, None]
            Phi = None if eigen is None else eigen.get(z)
            if Phi is not None and np.size(Phi):
                Phi  = np.asarray(Phi, dtype=float)
                f    = np.ones(Phi.shape[1]) if shrink is None else np.asarray(
                    shrink.get(z, np.ones(Phi.shape[1])), dtype=float
                )
                keep = np.sqrt(np.clip(1.0 - f, 0.0, 1.0))
                proj = Phi.T @ (wq[:, None] * res)
                res  = res - Phi @ ((1.0 - keep)[:, None] * proj)

            #--Per-season level, in the same L2(w) inner product, taken after the
            #--eigen deflation so the two do not claim the same variation. Removed
            #--from the residuals for exactly the reason the eigenfunctions are:
            #--the historical likelihood now carries it as a rank-1 delta_sd^2 11'
            #--term, so leaving it in would have sigma(tau) describe it twice.
            off = wn @ res
            offsets.append(off)
            res = res - off[None, :]
            pieces.append(res)
        R = None
        if pieces:
            R      = np.concatenate(pieces, axis=1)
            emp_sd = np.nanstd(R, axis=1)
            n_res  = int(R.shape[1])
        else:
            emp_sd = np.ones(self.T)
            n_res  = 1

        self.delta_sd = self._level_prior_sd(offsets, emp_sd)

        self.sigma_ell = self._fit_length_scale(R)
        #--Floors are relative to the data, not an absolute 1.0 case. The runner
        #--standardizes each location by its own mean and sd before pivoting
        #--(scale_data in run_real-time_cluster_stat2.py), so Y arrives with sd ~1
        #--while the real within-cluster residual sd is ~0.03. An absolute floor of
        #--1.0 pinned emp_sd at 1.0 at every tau, putting the prior center at
        #--sigma = 1 -- noise as wide as the entire signal -- and the posterior had
        #--to drive sigma_z to -7..-10 to climb back out of it.
        yvals = [np.asarray(v, float).ravel() for v in clustered.values() if v.shape[1]]
        y_sd  = float(np.nanstd(np.concatenate(yvals))) if yvals else 1.0
        if not np.isfinite(y_sd) or y_sd <= 0:
            y_sd = 1.0
        floor = 1e-2 * y_sd

        emp_sd = np.where(np.isfinite(emp_sd), emp_sd, floor)
        emp_sd = np.maximum(emp_sd, floor)

        #--No pre-smoothing and no upper clip. A 5-point moving average plus a
        #--clip at the 95th percentile made sense when sigma was linear
        #--interpolation between 5 knots, which could not smooth itself. The
        #--P-spline below smooths with lambda chosen by GCV, so doing it twice only
        #--shaves the peak: it cut the dynamic range of sigma(tau) from ~110x to
        #--~70x and moved the prior center a factor of 4 off the data.
        #--Observation noise, estimated from the high-frequency content of the
        #--series rather than from emp_sd. For a smooth curve plus white noise the
        #--second differences of the noise have variance 6 sigma^2, so
        #--sd(diff(y,2))/sqrt(6) measures the week-to-week reporting jitter without
        #--having to model the trend. Calendar-time columns, before the
        #--disease-time interpolation smooths them.
        #--This matters more than it looks. sigma(tau) is a function of disease
        #--time, and `peak` sets which tau the observed weeks land on, so if the
        #--nugget is too small to cover the jitter the only way the model can widen
        #--the current season is to slide `peak` until the observations sit on the
        #--high-variance part of sigma(tau) -- i.e. declare the season already
        #--peaked. A nugget sized to the actual jitter removes that incentive.
        Yc        = np.asarray(getattr(self, "Yscaled", self.Y), dtype=float)
        d2        = np.diff(Yc, n=2, axis=0)
        obs_noise = float(np.nanmedian(np.nanstd(d2, axis=0))) / np.sqrt(6.0)
        if not np.isfinite(obs_noise) or obs_noise <= 0:
            obs_noise = floor

        self.sigma_y_sd         = y_sd
        self.sigma_floor        = floor
        self.sigma_emp_sd       = emp_sd
        #--A ceiling at 2x the empirical max was a modelling constraint, not a
        #--safety rail: the posterior sat flat against it for a third of the season
        #--(sigma pinned at 54 admissions from week 4 to week 20). Put it far enough
        #--out that only a genuinely runaway sigma reaches it. 3x the spread of Y is
        #--already past the point where sigma exceeds the whole signal.
        self.sigma_max          = float(max(emp_sd.max() * 10.0, 3.0 * y_sd))
        self.sigma_obs_noise    = obs_noise
        self.sigma_nugget_scale = float(max(obs_noise, floor))
        self.sigma_n_res        = n_res

        #--Prior centre for the nugget, on the logit scale because that is where it
        #--is sampled. Clipped off both ends: a fraction of 0 means no reporting
        #--noise at all and a near-singular K_OO, and anything past 0.6 means the
        #--season is mostly noise and the GP carries no signal between weeks.
        eta = float(np.clip(self.sigma_nugget_frac, 0.02, 0.60))
        self.sigma_nugget_frac  = eta
        self.sigma_nugget_logit = float(np.log(eta / (1.0 - eta)))

        self._make_bspline_knots(
            weeks_per_knot=self.SIGMA_WEEKS_PER_KNOT, degree=3
        )
        B         = self._bspline_design(self.disease_grid)
        n_basis   = self.sigma_n_basis
        self.sigma_B_grid = B

        P   = self._diff_matrix(n_basis, order=2)
        PtP = P.T @ P
        y   = np.log(np.maximum(emp_sd, floor))
        I   = np.eye(n_basis)

        #--Smoothing parameter by GCV rather than a guess, so the penalty matches
        #--how much structure the empirical SD actually carries.
        best = None
        for lam in np.logspace(-4, 6, 41):
            A   = B.T @ B + lam * PtP + 1e-10 * I
            Hat = B @ np.linalg.solve(A, B.T)
            r   = y - Hat @ y
            dof = float(np.trace(Hat))
            den = max(self.T - dof, 1e-6)
            gcv = self.T * float(r @ r) / (den ** 2)
            if best is None or gcv < best[0]:
                best = (gcv, float(lam))
        lam = best[1]
        self.sigma_lambda = lam

        #--Prior mean for beta: the penalized fit of the empirical log SD, so the
        #--curve starts from a real case-scale SD and not free basis weights.
        A_lam              = B.T @ B + lam * PtP + 1e-10 * I
        beta_hat           = np.linalg.solve(A_lam, B.T @ y)
        self.sigma_beta_hat = beta_hat

        #--Prior width on log sigma. Two things make the center uncertain: emp_sd
        #--is a sample SD, worth 1/sqrt(2(n-1)) in log units, and the smooth spline
        #--cannot follow every wiggle of the empirical curve, worth the residual
        #--spread of the fit. Using the sampling term alone (the earlier version)
        #--assumes the center is right and gave 0.18, which the likelihood then had
        #--to break by 7-11 sd to reach the sigma it actually wanted.
        resid                = y - B @ beta_hat
        self.sigma_log_scale = float(np.clip(
            np.sqrt(1.0 / (2.0 * max(n_res - 1, 1)) + float(np.var(resid))),
            0.20, 1.50,
        ))

        #--Whiten the penalty once, here, instead of sampling a random walk in the
        #--model. prior precision = lam * P'P + ridge * I, so a standard normal z
        #--maps to a penalized beta through cholesky(cov). The ridge is what keeps
        #--the covariance finite on the penalty's null space (the constant and
        #--linear terms a second-difference penalty cannot see).
        ridge = 1e-3 * max(lam, 1.0)
        cov   = np.linalg.inv(lam * PtP + ridge * I)
        cov   = 0.5 * (cov + cov.T)
        L     = np.linalg.cholesky(cov + 1e-12 * I)

        #--Rescale so the marginal prior sd of log sigma on the grid equals the
        #--sampling width above. Without this the ridge sets the level's spread
        #--arbitrarily.
        marg  = float(np.mean(np.sqrt(np.maximum(np.diag(B @ cov @ B.T), 1e-18))))
        L     = L * (self.sigma_log_scale / max(marg, 1e-12))

        #--Precomputed so the model is one matmul: no spline evaluation, no
        #--cumulative sums, no hierarchical scale multiplying a unit normal.
        self.sigma_logsd_grid = B @ beta_hat
        self.sigma_M          = B @ L

    def _diff_matrix(self, n, order=2):
        """order-th difference operator, shape (n - order, n)."""
        Dm = np.eye(n)
        for _ in range(order):
            Dm = np.diff(Dm, axis=0)
        return Dm

    def _make_bspline_knots(self, weeks_per_knot=5, degree=3):
        """Clamped cubic knots, equally spaced about every `weeks_per_knot` weeks.

        Interior knots are uniform in tau, which is what a P-spline needs: the
        difference penalty treats neighbouring coefficients as comparable, and
        that only holds when the knots they sit on are a constant distance apart.
        """
        lo = float(self.disease_grid.min())
        hi = float(self.disease_grid.max())
        if hi <= lo:
            hi = lo + 1.0

        #--tau in [0, 2] spans T weeks, so a span is 2 * weeks_per_knot / T wide.
        n_span   = max(int(round(self.T / float(weeks_per_knot))), 2)
        internal = np.linspace(lo, hi, n_span + 1)[1:-1]

        #--A second-difference penalty needs at least 3 coefficients to act on.
        degree = int(degree)
        while len(internal) + degree + 1 < 4 and degree > 1:
            degree -= 1

        self.sigma_degree   = degree
        self.sigma_knots    = np.concatenate(
            [np.full(degree + 1, lo), internal, np.full(degree + 1, hi)]
        )
        self.sigma_n_basis  = len(internal) + degree + 1
        self.sigma_knot_tau = internal
        self.weeks_per_knot = float(self.T) / n_span

    def _bspline_design(self, x):
        x = np.asarray(x, dtype=float)
        return np.asarray(
            BSpline.design_matrix(x, self.sigma_knots, self.sigma_degree).todense(),
            dtype=float,
        )

    def compute_covariance_and_eigenfunction(self, y, mean_t=None):
        T      = self.T
        Ys     = y.T
        S_past = Ys.shape[0]
        
        mu     = np.asarray(self.mean_t if mean_t is None else mean_t, dtype=float)
        X      = np.column_stack([np.ones(T), mu])

        es = np.zeros((T, S_past), dtype=float)
        for s, ys in enumerate(Ys):
            coef, _, _, _ = np.linalg.lstsq(X, ys, rcond=None)
            es[:, s] = ys - X @ coef

        C = (1.0 / max(S_past - 1, 1)) * es @ es.T

        w         = self.delta_tau * np.array([0.5] + [1] * (T - 2) + [0.5])
        Whalf     = np.diag(np.sqrt(w))
        Whalf_inv = np.diag(1.0 / np.sqrt(w))

        lambdas, eigenfuncs = np.linalg.eigh(Whalf @ C @ Whalf)
        eigenfuncs          = Whalf_inv @ eigenfuncs

        lambdas     = lambdas[::-1]
        eigenfuncs  = eigenfuncs[:, ::-1]
        return es, lambdas, eigenfuncs

    def compute_mean_func(self, Y, grid):
        ys_disease_time = []
        for i, peak in enumerate(self.past_peaks):
            h = 1 + (self.calendar_grid - peak) / self.T
            ys_disease_time.append(np.interp(grid, h, Y[:, i]))
        self.Ys_disease_time = np.array(ys_disease_time)
        return np.mean(ys_disease_time, axis=0)

    def compute_peaks(self, Y):
        return np.nanargmax(Y[:, :-1], axis=0)

    def target_location_past_idx(self):
        """Past-column indices whose location matches the series being forecast.

        `D` columns are (season, location); the last column is the current
        (season, location). Indices are into the past block `D.iloc[:, :-1]`.
        """
        D = self.D
        n_past = D.shape[1] - 1
        if n_past < 1:
            return np.array([], dtype=int)
        if not isinstance(D, pd.DataFrame):
            return np.arange(n_past)
        last = D.columns[-1]
        if not (isinstance(last, tuple) and len(last) >= 2):
            return np.arange(n_past)
        loc = last[1]
        return np.array(
            [
                i
                for i, c in enumerate(D.columns[:-1])
                if isinstance(c, tuple) and len(c) >= 2 and c[1] == loc
            ],
            dtype=int,
        )

    def _peaks_of(self, Y, idx):
        """Calendar peak week of each *completed* column in `idx`.

        Only the target's own current season is the last column of D; the donors'
        current seasons sit in the middle of the past block holding t0 weeks of a
        season that has not peaked, so a raw argmax over them returns week 1 or 3
        and pooling those in takes the donor peak-week sds from 0.5-1.5 to 6.4-7.4.
        drop_partial_past removes those columns before anything here runs, so this
        is a second line of defence rather than the fix.
        """
        cols = np.asarray(Y, dtype=float)[:, idx] if len(idx) else np.zeros((Y.shape[0], 0))
        if cols.size == 0:
            return np.zeros(0)
        keep = np.isfinite(cols).sum(axis=0) >= self.MIN_SEASON_COVERAGE * cols.shape[0]
        cols = cols[:, keep]
        if cols.shape[1] == 0:
            return np.zeros(0)
        return np.nanargmax(cols, axis=0).astype(float)

    def compute_peak_prior(self, Y):
        """Calendar-time peak mean and *predictive* sd for the current season.

        The sd was previously the raw sample sd of the target's own past peak
        weeks, floored at one week. Two things were wrong with that.

        First, four seasons is not enough to pin a between-season sd. Across the
        52 locations in the target file the peak-week sd has median 4.11 weeks and
        an interquartile range of 2.96 to 4.57, so locations are close to
        exchangeable in this respect -- but Hawaii's four seasons peaked in weeks
        17, 18, 18, 18, giving a sample sd of 0.50, the smallest of all 52. Taken
        at face value and floored to 1.0 that is a very tight, very late prior.
        The posterior then had to fight it 4.4 sd to get the peak earlier, could
        not get far enough, and the amplitude exploded to 13 instead -- a 4.25x
        forecast overshoot. A floor cannot help here: only 3 of 52 locations fall
        below 1.0 week, so the floor binds almost nowhere while the real problem
        is a badly estimated sd being trusted.

        So pool. Shrink the location's own variance toward the pooled variance of
        every location in D, weighting by degrees of freedom, which leaves a
        well-estimated location almost untouched and pulls a pathological one
        back to the population.

        Second, this is a prior for a season that has not happened, while the mean
        it is centred on is itself estimated from n seasons. The predictive
        variance is therefore s^2 (1 + 1/n), not s^2.
        """
        past = np.asarray(Y[:, :-1], dtype=float)
        idx  = self.target_location_past_idx()

        own    = self._peaks_of(past, idx if idx.size else np.arange(past.shape[1]))
        n_own  = own.size
        floor = self.PEAK_SD_FLOOR_FRAC * self.T
        if n_own == 0:
            return float(self.T) / 2.0, float(floor)

        peak_mean = float(np.mean(own))
        var_own   = float(np.var(own, ddof=1)) if n_own > 1 else np.nan
        df_own    = max(n_own - 1, 0)

        #--Pooled across every location in D, target and donors alike.
        num = den = 0.0
        for cols in self._past_idx_by_location().values():
            pk = self._peaks_of(past, cols)
            if pk.size > 1:
                num += (pk.size - 1) * float(np.var(pk, ddof=1))
                den += pk.size - 1
        var_pool = num / den if den > 0 else np.nan

        if not np.isfinite(var_own) and not np.isfinite(var_pool):
            var = floor ** 2
        elif not np.isfinite(var_own):
            var = var_pool
        elif not np.isfinite(var_pool):
            var = var_own
        else:
            nu  = self.PEAK_SD_POOL_DF
            var = (df_own * var_own + nu * var_pool) / max(df_own + nu, 1e-9)

        #--Predictive, not descriptive: the centre is estimated from n_own seasons.
        var *= 1.0 + 1.0 / max(n_own, 1)

        self.peak_sd_own  = float(np.sqrt(var_own)) if np.isfinite(var_own) else np.nan
        self.peak_sd_pool = float(np.sqrt(var_pool)) if np.isfinite(var_pool) else np.nan

        return peak_mean, float(max(np.sqrt(var), floor))

    def _past_idx_by_location(self):
        """Past-column indices grouped by location, for pooling across donors.

        Falls back to one pooled group when D has no (season, location) columns,
        which is how the standalone tests drive this class.
        """
        D      = self.D
        n_past = D.shape[1] - 1
        if n_past < 1:
            return {}
        if not isinstance(D, pd.DataFrame):
            return {"_all": np.arange(n_past)}

        by = {}
        for i, c in enumerate(D.columns[:n_past]):
            key = c[1] if isinstance(c, tuple) and len(c) >= 2 else "_all"
            by.setdefault(key, []).append(i)
        return {k: np.array(v, dtype=int) for k, v in by.items()}

    def compute_peak_intensities(self, Y):
        return np.nanmax(Y[:, :-1], axis=0)

    def scale(self, Y):
        Y = np.asarray(Y, dtype=float)

        #self.global_mean = np.nanmean( np.nanmean(Y,axis=0))
        #self.global_std  = np.nanmean( np.nanstd(Y,axis=0))

        self.global_mean = 0.0
        self.global_std  = 1.0
        
        return (Y - self.global_mean) / self.global_std

    def drop_partial_past(self, D):
        """Drop past columns that do not cover enough of their season to be one.

        Only the *target's* current season is the last column of D. The donor
        locations' current seasons sit inside the past block holding t0 weeks with
        the rest missing, and interp_Y fills across columns within a location, so
        those 40 missing weeks were carried over from the donor's own previous
        season -- exactly, correlation 1.0000 and a difference of identically 0
        past week t0. Three of Montana's nineteen "historical seasons" were
        therefore verbatim copies of 2025/2026, which entered that one season four
        times over into the cluster means, the eigenfunctions and sigma(tau).

        A four-week fragment is unusable by any of the complete-curve machinery
        downstream, so drop it rather than invent the remainder. Dropping happens
        before self.D is stored so that the column labels stay aligned with the
        numeric array: target_location_past_idx and _past_idx_by_location both
        index positionally into D.columns.
        """
        arr    = np.asarray(getattr(D, "values", D), dtype=float)
        n_past = arr.shape[1] - 1
        if n_past < 1:
            return D
        cover = np.isfinite(arr[:, :n_past]).sum(axis=0)
        keep  = cover >= self.MIN_SEASON_COVERAGE * arr.shape[0]
        if keep.all():
            return D
        idx = np.append(np.flatnonzero(keep), arr.shape[1] - 1)
        return D.iloc[:, idx] if isinstance(D, pd.DataFrame) else D[:, idx]

    def interp_Y(self, D):
        target = D.iloc[:, -1]
        past   = D.iloc[:, :-1]

        #--interpolate(axis=1) fills a missing week from the neighbouring columns.
        #--Once D carries (season, location) columns those neighbours are other
        #--states, so a state's gap gets filled from series on a different scale
        #--(week 53 exists in one season only, which pulled every column toward it).
        #--Group by location so a season can only borrow from its own location.
        if isinstance(past.columns, pd.MultiIndex) and past.columns.nlevels >= 2:
            locs   = past.columns.get_level_values(1)
            filled = past.copy()
            for loc in locs.unique():
                block = past.loc[:, locs == loc]
                filled.loc[:, block.columns] = block.interpolate(
                    axis=1, limit_direction="both"
                )
        else:
            filled = past.interpolate(axis=1, limit_direction="both")

        return np.hstack([filled, target.values[:, None]])


#--Drop-in alias so an existing runner only needs its import line changed.
cluster_stat = cluster_stat3

##--MODELEND

