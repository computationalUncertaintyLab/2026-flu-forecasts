#mcandrew
#
# puca_shapev2 -- a DERIVATIVE (functional-ARIMA) shape-library forecaster.
#
# ============================ V2 vs V1 ================================
# V1 (puca_shape) fits the LEVEL y(t) and forecasts by integrating a derivative
# (a fit/forecast mismatch). V2 fits the WEEKLY CHANGE (first difference) --
# functional-ARIMA(d=1): the observation model is on Delta y(t), so the absolute
# level is differenced away and is NOT a fit latent. Concretely
#     Delta y(t) = mu_drift + amplitude * [ f_p( h(t+1) ) - f_p( h(t) ) ] + err(t)
# marginalized over the analog shape p (categorical, as in V1). mu_drift is the
# derivative-space "intercept" (a constant weekly drift = a linear term in level
# space). Consequences:
#   * The single derivative likelihood REPLACES V1's two layers (level toe
#     likelihood + slope term). No more double level+derivative penalty.
#   * The level intercept is differenced out; height is set only by the forecast
#     anchor (last observed value) + the external peak_y/total_y constraints,
#     which are LEVEL-space and so are applied to the INTEGRATED (reconstructed)
#     curve here (not folded into the amplitude prior as in V1 CHANGE (P)).
#   * Forecast = the EXACT antiderivative of the fitted derivative, i.e. the
#     original level spline f_p (SHAPE_LIB) anchored at the last observed value
#     (no numerical integration), + the bridge.
# Risk (inherent to differencing): integration accumulates error, so level
# uncertainty grows with horizon and the peak height leans harder on peak_y.
# =====================================================================
#
# Motivation
# ----------
# The SIR-based `puca` spends its effort integrating an epidemic ODE whose
# dynamics are only weakly identified from the ~5 observed weeks of the target
# season -- the real work of the target forecast is done by the external
# constraints (peak time, peak height, season total). `puca_shape` drops the
# ODE entirely and instead uses a POOLED LIBRARY of past-season shapes,
# represented by B-spline coefficients (see
# analysis/normalize_and_derive_B_weights.py): each past (location, season) is
# normalized to unit peak, aligned to peak at tau=1, and summarized by its spline
# coefficients. The target curve picks ONE analog shape from this library
# (categorical, marginalized during inference and drawn forward at forecast
# time), warped in time so its peak lands at a latent peak location, then scaled
# to a peak height (the season's width comes from the chosen shape, not a
# separate latent). Constraints enter as soft penalties on the realized
# functionals (peak time / peak height / season total), and the observed toe
# weeks enter through a Gaussian likelihood.
#
# Consequences
#   * No `lax.scan` epidemic integration  -> ~orders of magnitude faster.
#   * No SIR R0/I0 ridge, no beta funnels  -> clean geometry, no divergences.
#   * Interpretable: "this year looks like a mix of 2022/23 and 2024/25,
#     a bit later and taller."
#   * Honest about partial observability: with no data, it reverts to the
#     climatological mixture of past shapes.
#
# Interface parity with `puca`
#   puca_shape(y=[...], Y=[...], X=None, anchor=None).fit(
#       total_y_target=, total_y_sd=, peak_time_target=, peak_time_sd=,
#       peak_y_target=, peak_y_sd=, peak_temperature=, peak_a_to_b=)
#   ...then .forecast() -> (draws, T) array on the raw case scale.
#
# NOTE: peak_temperature is accepted for signature parity but unused (peak
# time/location are analytic in this parameterization).
#
# CHANGE (B): if peak_a_to_b (a per-location DataFrame with b0/b1/sd) is passed,
# a second influenza-B wave is added: the SAME chosen analog shape, placed at
# peak_b = b0 + b1*peak_a (+/- sd). The A wave is the reference (weight fixed to
# 1) and B carries a RELATIVE weight w_b, so trend = intercept + amplitude*(
# shape_A + w_b*shape_B). Pass peak_a_to_b=None to revert to the A-only model.
#
# CHANGE (D): at forecast time, optionally integrate the analog shape DERIVATIVE
# forward from the last observed level instead of using the analog LEVEL curve.
# Fitting still uses the level likelihood + categorical marginalization. Set
# FORECAST_FROM_DERIVATIVE=False (or forecast_from_derivative=False in forecast())
# to revert to level-based forecasts.
#
# CHANGE (S): the categorical shape score adds a SLOPE-matching term -- each
# analog is scored on how well its recent weekly change matches the observed
# weekly change, so the shape weights (and sampled forecast) prefer analogs
# currently moving like us. Set SLOPE_LIKELIHOOD=False to revert.
#
# CHANGE (P): the peak-height constraint is folded into the amplitude PRIOR --
# amplitude ~ TruncatedNormal(peak_y_target - toe_loc, peak_y_sd, low=0) so that
# peak_y = intercept + amplitude ~ peak_y_target. This replaces the old
# HalfNormal(amplitude_scale) prior (mode at 0, which let the peak collapse to
# the current level on a rising toe) AND the separate peak_y penalty (removed),
# so the peak-height info is counted exactly once.
#
# CHANGE (E): HETEROSCEDASTIC BRIDGED PROCESS ERROR. The iid observation noise
# sigma does not accumulate, so multi-step forecast intervals were too narrow. A
# forward random walk makes uncertainty largest at season end, where flu is
# usually known to be near baseline; we instead use a random-walk BRIDGE e_t on
# the horizon (e=0 at the last observed week and e=0 at the final season week).
# The per-week innovation is HETEROSCEDASTIC -- its sd scales with the local
# forecast level, s_t = PROCESS_SD_FRAC * max(trend_fc(t), 0) -- so the fan is
# large near the peak and small on the flat parts. It is deliberately NOT fit
# from the flat, uninformative toe. The predictive model is
# y_pred ~ N(trend_fc + e_t, sigma). Set PROCESS_ERROR False to revert.

import os

import numpy as np
import pandas as pd

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

from numpyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO, init_to_median

from patsy import dmatrix


#--B-spline basis config. MUST match analysis/normalize_and_derive_B_weights.py
#--EXACTLY (coarse uniform backbone over [0,2] + dense cluster around the peak
#--tau=1, degree 3) so the saved coefficients evaluate to the same shapes here.
_SPLINE_DEGREE    = 3
_TAU_LOWER        = 0.0
_TAU_UPPER        = 2.0
_N_COARSE_KNOTS   = 10
_PEAK_CLUSTER     = np.array([0.82, 0.88, 0.94, 1.0, 1.06, 1.12, 1.18])
_coarse           = np.linspace(_TAU_LOWER, _TAU_UPPER, _N_COARSE_KNOTS + 2)[1:-1]
_INTERIOR_KNOTS   = np.unique(np.round(np.concatenate([_coarse, _PEAK_CLUSTER]), 6))
_N_INTERIOR_KNOTS = _INTERIOR_KNOTS.size


def _spline_basis(tau):
    """Cubic B-spline design matrix on tau, identical to the derive script."""
    B = dmatrix(
        "bs(x, knots=knots, degree=degree, lower_bound=lo, upper_bound=hi, include_intercept=False) - 1",
        {"x": np.asarray(tau, float), "knots": _INTERIOR_KNOTS,
         "degree": _SPLINE_DEGREE, "lo": _TAU_LOWER, "hi": _TAU_UPPER},
    )
    return np.asarray(B)


def _spline_basis_deriv(tau, h=1e-3):
    """First derivative dB/dtau, matching analysis/normalize_and_derive_B_weights.py."""
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    xc  = np.clip(tau, _TAU_LOWER + h, _TAU_UPPER - h)
    return (_spline_basis(xc + h) - _spline_basis(xc - h)) / (2.0 * h)


def _integrate_forecast_from_deriv(y_anchor, weekly_change, tobs):
    """
    Integrate weekly changes forward from the last observed week.

    weekly_change[t] is the increment from calendar week t to t+1. The path is
    anchored at y_anchor = y[tobs-1].
    """
    T = weekly_change.shape[0]
    if tobs >= T:
        return jnp.full((T,), y_anchor)

    future_idx = jnp.arange(tobs, T)
    _, future_levels = jax.lax.scan(
        lambda y_prev, t: (y_prev + weekly_change[t - 1], y_prev + weekly_change[t - 1]),
        y_anchor,
        future_idx,
    )

    trend_fc = jnp.full((T,), jnp.nan)
    trend_fc = trend_fc.at[:tobs].set(jnp.nan)          #--filled below from observed y
    trend_fc = trend_fc.at[tobs - 1].set(y_anchor)
    trend_fc = trend_fc.at[tobs:].set(future_levels)
    return trend_fc


class puca_shapev2(object):

    #--observation-noise floor as a fraction of the amplitude scale, so sigma
    #--cannot collapse to ~0 and make the toe likelihood explode.
    SIGMA_FLOOR_FRAC = 0.02

    #--CHANGE (B): HalfNormal scale for w_b, the B weight relative to A (whose
    #--weight is fixed to 1). The B wave is typically a fraction of the A wave;
    #--0.5 keeps it mostly below the A peak.
    B_FRAC_SCALE = 0.5

    #--pooled B-spline coefficient library (all locations/seasons), produced by
    #--analysis/normalize_and_derive_B_weights.py. We read the *_weights file
    #--(coefficients + the season length `n`) so we can drop partially-observed
    #--seasons, whose "shape" is just a toe and would be a degenerate template.
    LIB_CSV          = "./analysis_data/normalized_B_weights.csv"
    MIN_SEASON_WEEKS = 40      #--exclude library seasons shorter than this (e.g. the in-progress target season)
    TAU_GRID_N       = 400     #--resolution of the fine tau grid the shapes are evaluated on

    #--CHANGE (D): forecast by integrating the analog shape DERIVATIVE forward from
    #--the last observed level, instead of reading off the analog LEVEL curve
    #--(which over-commits to historical peak heights). Set False to revert.
    FORECAST_FROM_DERIVATIVE = True

    #--CHANGE (S): slope-matching term in the categorical shape score. In addition
    #--to the observed LEVEL likelihood, each analog is scored on how well its
    #--weekly change (discrete derivative of its level trend) matches the recent
    #--observed weekly change. This makes the shape WEIGHTS (and thus the sampled
    #--forecast) prefer analogs currently moving like us. Set SLOPE_LIKELIHOOD
    #--False to revert.
    SLOPE_LIKELIHOOD = True
    SLOPE_N_WEEKS    = 4       #--use only the last N observed week-to-week changes
    SLOPE_SD_FRAC    = 0.25    #--slope-noise floor as a fraction of amplitude_scale (start weak)

    #--CHANGE (E): BRIDGED PROCESS ERROR on the forecast path. The observation
    #--noise sigma is iid and does NOT accumulate, so multi-step forecast
    #--intervals do not fan out. A one-sided forward random walk would make
    #--uncertainty largest at season end, but we usually know the end-of-season
    #--level is close to baseline. Instead we use a random-walk BRIDGE: e_t is
    #--zero at the last observed week and zero again at the final season week,
    #--with largest variance in the unobserved middle. The observed y_pred is
    #--   y_pred ~ N(trend_fc + e_t, sigma).
    #--The per-week innovation is HETEROSCEDASTIC: its sd scales with the LOCAL
    #--forecast level, s_t = PROCESS_SD_FRAC * max(trend_fc(t), 0). This is NOT
    #--fit from the (flat, uninformative) toe -- the toe carries no information
    #--about epidemic-phase deviation -- so the fan is large near the peak and
    #--small on the flat parts automatically. PROCESS_SD_FRAC is the per-week
    #--innovation as a fraction of the local level. Set PROCESS_ERROR False to revert.
    PROCESS_ERROR    = True    #--heteroscedastic random-walk bridge on the forecast path
    PROCESS_SD_FRAC  = 0.10    #--per-week innovation sd as a fraction of the LOCAL level

    def __init__(self, y=None, Y=None, X=None, anchor=None):
        self.X__input = X
        self.y__input = y
        self.Y__input = Y
        self.anchor   = anchor
        self.organize_data()

    # ------------------------------------------------------------------
    # Data prep
    # ------------------------------------------------------------------
    def organize_data(self):
        def interp_nans(col):
            col = np.asarray(col, float)
            idx = np.arange(len(col))
            good = np.isfinite(col)
            if good.all():
                return col
            col[~good] = np.interp(idx[~good], idx[good], col[good])
            return col

        y = np.asarray(self.y__input[0], float)     #--target season (raw, NaN future)
        Y = np.asarray(self.Y__input[0], float)     #--past seasons  (raw, T x P)

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        #--past seasons must be complete templates; interpolate any residual gaps.
        Y_complete = np.column_stack([interp_nans(Y[:, p]) for p in range(Y.shape[1])])

        T = Y_complete.shape[0]

        #--first NaN in the target marks the end of observed data.
        nan_idx = np.argwhere(~np.isfinite(y)).ravel()
        tobs    = int(nan_idx[0]) if nan_idx.size else T

        #--reference scale (kept for reporting / init; model works on raw scale).
        self.global_mu  = float(np.nanmean(np.nanmean(Y_complete, 0)))
        self.global_std = float(np.nanmean(np.nanstd(Y_complete, 0)))

        self.T    = T
        self.tobs = tobs
        self.y    = y               #--raw target (with NaNs beyond tobs)
        self.Y    = Y_complete      #--raw, complete past templates (T x P)
        self.X    = None
        self.P    = Y_complete.shape[1]

        return y, Y_complete, None

    # ------------------------------------------------------------------
    # Build the pooled shape library from saved B-spline coefficients
    # ------------------------------------------------------------------
    def _build_library(self):
        """
        Load the pooled B-spline coefficient library and evaluate each
        coefficient vector into a smooth, peak-aligned (peak at tau=1), unit-peak
        shape on a fine tau grid. These shapes are the discrete "analog seasons"
        the model chooses among (see `model`).
        """
        T = self.T

        coef_df = pd.read_csv(self.LIB_CSV)

        #--drop partially-observed seasons (their derived shape is just a toe)
        if "n" in coef_df.columns:
            coef_df = coef_df.loc[coef_df["n"] >= self.MIN_SEASON_WEEKS]

        bcols = [c for c in coef_df.columns if c.startswith("b") and c[1:].isdigit()]
        COEF  = coef_df[bcols].to_numpy(float)                    #--(P, K)

        #--evaluate coefficients -> smooth shapes on a fine tau grid, using the
        #--SAME basis the coefficients were fit with.
        tau_grid = np.linspace(_TAU_LOWER, _TAU_UPPER, self.TAU_GRID_N)
        Bfine    = _spline_basis(tau_grid)                       #--(G, K)
        Bdd      = _spline_basis_deriv(tau_grid)                 #--(G, K)
        shapes   = COEF @ Bfine.T                                #--(P, G)
        derivs   = COEF @ Bdd.T                                    #--(P, G)

        #--incidence shapes are non-negative; clip spline undershoot.
        shapes = np.clip(shapes, 0.0, None)

        #--Anchor each shape to unit height AT tau = 1 (the aligned peak), not the
        #--global max. The warp places tau = 1 at new_peak, so this makes peak_y =
        #--intercept + amplitude exact and the peak-time constraint valid, while
        #--preserving any (now closely-fit) secondary waves that may exceed 1.
        i1     = int(np.argmin(np.abs(tau_grid - 1.0)))
        v1     = shapes[:, i1]
        keep   = v1 > 1e-6                                        #--drop degenerate rows
        shapes = shapes[keep] / v1[keep, None]
        derivs = derivs[keep] / v1[keep, None]                    #--same scale as level shape

        self.SHAPE_LIB     = jnp.asarray(shapes)                 #--(P, G)
        self.DERIV_LIB     = jnp.asarray(derivs)                 #--(P, G) CHANGE (D)
        self.tau_grid      = jnp.asarray(tau_grid.astype(float)) #--(G,)
        self.calendar_time = jnp.asarray(np.arange(T).astype(float))
        self.P_lib         = int(shapes.shape[0])

        #--climatology of peak weeks from the LOCAL past seasons (Y), used only
        #--for the new_peak prior and amplitude/toe scale heuristics.
        R = self.Y
        self.peak_week_past = np.nanargmax(R, axis=0)
        self.peak_val_past  = np.nanmax(R, axis=0)
        return shapes

    # ------------------------------------------------------------------
    # The probabilistic model
    # ------------------------------------------------------------------
    @staticmethod
    def model(y                = None,
              SHAPE_LIB        = None,
              DERIV_LIB        = None,      #--CHANGE (D)
              tau_grid         = None,
              calendar_time    = None,
              T                = None,
              tobs             = None,        #--CHANGE (D): last observed week index
              mean_peak        = None,
              peak_spread      = None,
              toe_loc          = None,
              toe_scale        = None,
              amplitude_scale  = None,
              sigma_scale      = None,
              sigma_floor      = None,
              peak_time_target = None,
              peak_time_sd     = None,
              peak_y_target    = None,
              peak_y_sd        = None,
              total_y_target   = None,
              total_y_sd       = None,
              peak_b_b0        = None,      #--CHANGE (B): peak_b = b0 + b1*peak_a (+/- sd)
              peak_b_b1        = None,
              peak_b_sd        = None,
              b_frac_scale     = 0.5,
              forecast_from_derivative = False,  #--CHANGE (D): unused in V2 (always integrate)
              drift_scale      = None,      #--V2: prior sd for the derivative-space drift mu_drift
              dy_obs           = None,      #--V2: observed weekly changes (T-1,)
              dobs_mask        = None,      #--V2: which weekly changes are observed (T-1,)
              slope_mask       = None,      #--(legacy, unused in V2)
              slope_sd_floor   = None,      #--(legacy, unused in V2)
              process_error    = False,     #--CHANGE (E): add bridged forecast-path error
              process_sd_frac  = 0.10,      #--CHANGE (E): per-week innovation sd / local level
              forecast         = False):

        eps = 1e-6
        P   = SHAPE_LIB.shape[0]
        use_b = peak_b_b0 is not None      #--CHANGE (B): two-component (A+B) toggle

        #--Continuous latents, SHARED across every candidate analog shape.
        #--WHERE the peak falls (calendar week); climatology prior + peak-time
        #--constraint below.
        new_peak  = numpyro.sample("new_peak", dist.Normal(mean_peak, peak_spread))

        #==V2: DERIVATIVE-space latents ========================================
        #--We fit the weekly CHANGE, so the absolute level (V1's `intercept`) is
        #--differenced away and is NOT sampled here. amplitude scales the
        #--derivative shape; mu_drift is the derivative-space "intercept" -- a
        #--constant weekly drift (a linear term in level space). sigma is the
        #--error on the weekly change. peak_y is level-space and so is NOT folded
        #--into the amplitude prior here (it becomes a soft constraint on the
        #--integrated level below); amplitude keeps a weakly-informative HalfNormal.
        amplitude = numpyro.sample("amplitude", dist.HalfNormal(amplitude_scale))
        mu_drift  = numpyro.sample("mu_drift", dist.Normal(0.0, drift_scale))

        #--Observation noise on the weekly change (raw scale) with a floor.
        sigma = numpyro.deterministic(
            "sigma", sigma_floor + numpyro.sample("sigma_raw", dist.HalfNormal(sigma_scale)))
        #==V2 end ==============================================================

        #--Warp calendar time into tau so the peak (tau=1) lands at new_peak
        #--(tau = 1 + (t - new_peak)/T; the season WIDTH comes from the chosen
        #--shape). We fit weekly CHANGES, and the observed change Delta y(t) =
        #--y(t+1)-y(t) is the AVERAGE rate over [t, t+1] (a secant). By the
        #--midpoint rule that secant equals the analytic spline derivative f'
        #--evaluated at the interval MIDPOINT t+1/2 (second-order accurate, no
        #--half-step bias). So we evaluate f' (DERIV_LIB, the B-spline's closed-form
        #--derivative) on the T-1 midpoints -- using the spline derivative directly
        #--rather than re-differencing an interpolated level curve.
        t_mid       = calendar_time[:-1] + 0.5                               #--(T-1,) interval midpoints
        tau_a_mid   = 1.0 + (t_mid - new_peak) / T                           #--(T-1,)
        deriv_a_mid = jax.vmap(lambda d: jnp.interp(tau_a_mid, tau_grid, d))(DERIV_LIB)  #--(P, T-1)
        deriv_comb  = deriv_a_mid                                            #--(P, T-1), A weight == 1

        #==CHANGE (B) begin: add a second (influenza-B) component ============
        #--Place B via the A->B peak regression (peak_b = b0 + b1*new_peak +/- sd)
        #--reusing the SAME analog shape; w_b is B's weight RELATIVE to A(=1).
        if use_b:
            peak_b   = numpyro.sample("peak_b",
                                      dist.Normal(peak_b_b0 + peak_b_b1 * new_peak, peak_b_sd))
            w_b      = numpyro.sample("w_b", dist.HalfNormal(b_frac_scale))  #--B weight relative to A(=1)
            tau_b_mid   = 1.0 + (t_mid - peak_b) / T                         #--(T-1,)
            deriv_b_mid = jax.vmap(lambda d: jnp.interp(tau_b_mid, tau_grid, d))(DERIV_LIB)  #--(P, T-1)
            deriv_comb  = deriv_comb + w_b * deriv_b_mid                     #--A(1) + B(w_b)
            numpyro.deterministic("amp_b", amplitude * w_b)                 #--absolute B-peak height
        #==CHANGE (B) end ====================================================

        y_fill = jnp.nan_to_num(y, nan=0.0)

        #==V2: DERIVATIVE likelihood (functional ARIMA d=1) ====================
        #--Modeled weekly change dy_pred[p,t] = mu_drift + (amplitude/T)*f'_p(tau_mid).
        #--Since tau = 1 + (t-peak)/T has dtau/dt = 1/T, (amplitude/T)*f'(tau_mid) is
        #--the midpoint-rule value of the one-week secant -- i.e. the per-week
        #--increment. The forecast integrates this SAME object (cumsum), so fit and
        #--forecast agree exactly.
        dy_pred = mu_drift + (amplitude / T) * deriv_comb                    #--(P, T-1)

        #--Single likelihood on the observed weekly changes (all observed diffs).
        L = jnp.sum(
            jnp.where(dobs_mask[None, :],
                      dist.Normal(dy_pred, sigma).log_prob(dy_obs[None, :]),
                      0.0),
            axis=1)                                                           #--(P,)

        #--Reconstruct the implied LEVEL path per shape for the level-space
        #--constraints (peak_y, total_y) and the forecast. We do NOT numerically
        #--cumsum the fitted derivative: by the fundamental theorem of calculus the
        #--antiderivative of f' IS the original level curve f_p, and we HAVE its
        #--basis (SHAPE_LIB). So we warp the level curve onto integer weeks, add the
        #--drift's linear antiderivative (mu_drift*t), and anchor the difference at
        #--the last observed level. This is the EXACT integral of the fitted
        #--continuous derivative (no quadrature error).
        tau_a_full = 1.0 + (calendar_time - new_peak) / T                   #--(T,)
        level_a    = jax.vmap(lambda s: jnp.interp(tau_a_full, tau_grid, s))(SHAPE_LIB)  #--(P, T) f_p(tau_a(t))
        comb_level = level_a                                                 #--(P, T), A weight == 1
        if use_b:
            tau_b_full = 1.0 + (calendar_time - peak_b) / T                 #--(T,)
            level_b    = jax.vmap(lambda s: jnp.interp(tau_b_full, tau_grid, s))(SHAPE_LIB)
            comb_level = comb_level + w_b * level_b                         #--A(1) + B(w_b)
        model_level = amplitude * comb_level + mu_drift * calendar_time[None, :]  #--(P, T) up to a constant
        y_anchor    = y_fill[tobs - 1]
        level_future = y_anchor + (model_level[:, tobs:] - model_level[:, tobs - 1:tobs])  #--(P, T-tobs)
        level_obs   = jnp.broadcast_to(y_fill[:tobs], (P, tobs))            #--(P, tobs) observed levels
        level_p     = jnp.concatenate([level_obs, level_future], axis=1)    #--(P, T)

        peak_y_p  = jnp.max(level_p, axis=1)                                 #--(P,)
        total_y_p = jnp.sum(level_p, axis=1)                                 #--(P,)
        if total_y_target is not None:
            L = L + dist.Normal(total_y_target, total_y_sd).log_prob(total_y_p)
        if peak_y_target is not None:
            L = L + dist.Normal(peak_y_target, peak_y_sd).log_prob(peak_y_p)
        #==V2 end ==============================================================

        #--CATEGORICAL over analog seasons, MARGINALIZED for inference: sum over
        #--shapes (log-sum-exp) with a uniform prior. This keeps every latent
        #--continuous (works with plain NUTS and VI), and is the exact marginal of
        #--sampling one shape. The chosen shape is drawn forward at forecast time.
        log_prior = -jnp.log(P)
        numpyro.factor("shape_marginal", jax.nn.logsumexp(log_prior + L))

        #--Posterior responsibilities over shapes (interpretability + forecasting).
        resp = numpyro.deterministic("shape_weights", jax.nn.softmax(log_prior + L))  #--(P,)

        #--Responsibility-weighted reconstructed level curve + functionals (report).
        trend   = numpyro.deterministic("trend", resp @ level_p)             #--(T,)
        peak_y  = numpyro.deterministic("peak_y", jnp.sum(resp * peak_y_p))  #--(scalar)
        peak_t  = numpyro.deterministic("peak_time", new_peak)
        numpyro.deterministic("total_y", jnp.sum(resp * total_y_p))

        #--Shape-INDEPENDENT constraint: all shapes peak at tau=1 -> new_peak.
        if peak_time_target is not None:
            numpyro.sample("peak_time_penalty",
                           dist.Normal(peak_time_target, peak_time_sd), obs=peak_t)
        #--V2: peak_y / total_y are soft constraints on the INTEGRATED level (added
        #--to the per-shape score L above), since level is not a fit latent here.

        if forecast:
            #--Draw ONE analog season from the posterior responsibilities (discrete
            #--"pick one"). Its forecast level path IS the exact-integrated
            #--reconstruction (original level spline anchored at the last observed
            #--value), already built above as level_p -- no numerical integration.
            shape_idx = numpyro.sample("shape_idx", dist.Categorical(probs=resp))
            trend_fc  = level_p[shape_idx]                                    #--(T,) observed toe + antiderivative

            numpyro.deterministic("trend_fc", trend_fc)
            numpyro.deterministic("chosen_shape", shape_idx)

            #==CHANGE (E) begin: HETEROSCEDASTIC bridged process error ===========
            #--Random-walk bridge on the forecast horizon whose per-week innovation
            #--scales with the LOCAL forecast LEVEL (heteroscedastic): the true
            #--path can stray far from the analog near the peak, little on the flat
            #--parts. The increment at week t has sd s_t = process_sd_frac *
            #--max(trend_fc(t), 0); this is deliberately NOT fit from the (flat,
            #--uninformative) toe. Build the level-scaled walk W_h = sum_{i<=h}
            #--s_i z_i and condition it to return to zero at the final season week
            #--(e_h = W_h - (h/H) W_H). Thus e=0 at both known ends (last observed
            #--week and season end) with the largest spread in the interior.
            if process_error:
                n_time = calendar_time.shape[0]
                with numpyro.plate("proc_t", n_time):
                    z = numpyro.sample("proc_innov", dist.Normal(0.0, 1.0))     #--(T,)
                calendar_idx = jnp.arange(n_time)
                future_mask  = calendar_idx >= tobs
                s_t          = process_sd_frac * jnp.clip(trend_fc, 0.0)        #--(T,) local-level scale
                incr         = jnp.where(future_mask, s_t * z, 0.0)             #--level-scaled increments
                w_path       = jnp.cumsum(incr)
                H            = jnp.maximum(n_time - tobs, 1)
                h            = jnp.clip(calendar_idx - tobs + 1, 0, H)
                e_path       = jnp.where(future_mask,
                                         w_path - (h / H) * w_path[-1],
                                         0.0)
                numpyro.deterministic("proc_error", e_path)
            else:
                e_path = jnp.zeros_like(trend_fc)
            #==CHANGE (E) end ====================================================

            fc = numpyro.sample("forecast", dist.Normal(trend_fc + e_path, sigma))
            numpyro.deterministic("y_pred", fc)

    # ------------------------------------------------------------------
    # Assemble the concrete model kwargs (data-driven priors -> constants)
    # ------------------------------------------------------------------
    def _model_kwargs(self, peak_time_target, peak_time_sd, peak_y_target,
                      peak_y_sd, total_y_target, total_y_sd, forecast,
                      forecast_from_derivative=None):
        self._build_library()

        y   = self.y
        Y   = self.Y
        obs = y[np.isfinite(y)]

        if forecast_from_derivative is None:
            forecast_from_derivative = bool(
                forecast and getattr(self, "forecast_from_derivative",
                                     self.FORECAST_FROM_DERIVATIVE))

        #--climatology of past peak weeks
        mean_peak   = float(np.mean(self.peak_week_past))
        peak_spread = float(np.std(self.peak_week_past) + 3.0)   #--never too tight

        #--baseline (toe) from the earliest observed weeks
        ntoe     = max(1, min(3, self.tobs))
        toe_loc  = float(np.nanmean(y[:ntoe]))
        toe_scale = float(np.nanstd(Y[:ntoe, :]) + 0.1 * (np.nanmax(Y) - np.nanmin(Y)) + 1.0)

        #--amplitude / noise scales
        amplitude_scale = float(peak_y_target if peak_y_target is not None else np.nanmax(Y))
        sigma_scale     = float(0.10 * amplitude_scale + 1.0)
        sigma_floor     = float(self.SIGMA_FLOOR_FRAC * amplitude_scale + 1e-6)

        #--V2: prior sd for the derivative-space drift mu_drift. A weakly-informative
        #--scale on the order of a typical weekly change (amplitude spread over the
        #--season), so the drift can absorb a mild linear trend without dominating.
        drift_scale = float(amplitude_scale / max(self.T, 1) + 1.0)

        #--CHANGE (B): A->B peak regression (peak_b = b0 + b1*peak_a, +/- sd). When
        #--self.peak_a_to_b is None (or NaN) the second component is disabled and
        #--the model is the original A-only shape model.
        peak_b_b0 = peak_b_b1 = peak_b_sd = None
        p2b = getattr(self, "peak_a_to_b", None)
        if p2b is not None:
            b0 = float(np.asarray(p2b["b0"]).ravel()[0])
            b1 = float(np.asarray(p2b["b1"]).ravel()[0])
            sd = float(np.asarray(p2b["sd"]).ravel()[0])
            if np.all(np.isfinite([b0, b1, sd])):
                peak_b_b0, peak_b_b1, peak_b_sd = b0, b1, sd

        #--V2: observed weekly changes ARE the likelihood (not a side term). Score
        #--ALL observed diffs. dy_obs / dobs_mask have length T-1 (change t -> t+1).
        dy_full   = np.diff(y)                                       #--(T-1,), NaN across future/gaps
        dobs_mask = np.isfinite(dy_full)
        dy_obs    = jnp.asarray(np.nan_to_num(dy_full, nan=0.0), float)
        dobs_mask = jnp.asarray(dobs_mask)
        slope_mask = slope_sd_floor = None                          #--legacy, unused in V2

        return dict(
            y                = jnp.asarray(y, float),
            SHAPE_LIB        = self.SHAPE_LIB,
            DERIV_LIB        = self.DERIV_LIB,              #--CHANGE (D)
            tau_grid         = self.tau_grid,
            calendar_time    = self.calendar_time,
            T                = float(self.T),
            tobs             = int(self.tobs),              #--CHANGE (D)
            mean_peak        = mean_peak,
            peak_spread      = peak_spread,
            toe_loc          = toe_loc,
            toe_scale        = toe_scale,
            amplitude_scale  = amplitude_scale,
            sigma_scale      = sigma_scale,
            sigma_floor      = sigma_floor,
            peak_time_target = peak_time_target,
            peak_time_sd     = peak_time_sd,
            peak_y_target    = peak_y_target,
            peak_y_sd        = peak_y_sd,
            total_y_target   = total_y_target,
            total_y_sd       = total_y_sd,
            peak_b_b0        = peak_b_b0,        #--CHANGE (B)
            peak_b_b1        = peak_b_b1,        #--CHANGE (B)
            peak_b_sd        = peak_b_sd,        #--CHANGE (B)
            b_frac_scale     = self.B_FRAC_SCALE,#--CHANGE (B)
            forecast_from_derivative = forecast_from_derivative,  #--CHANGE (D) (unused in V2)
            drift_scale      = drift_scale,      #--V2
            dy_obs           = dy_obs,           #--V2
            dobs_mask        = dobs_mask,        #--V2
            slope_mask       = slope_mask,       #--(legacy)
            slope_sd_floor   = slope_sd_floor,   #--(legacy)
            process_error    = bool(getattr(self, "process_error", self.PROCESS_ERROR)),  #--CHANGE (E)
            process_sd_frac  = float(self.PROCESS_SD_FRAC),   #--CHANGE (E)
            forecast         = forecast,
        )

    # ------------------------------------------------------------------
    # NUTS fit
    # ------------------------------------------------------------------
    def fit(self,
            peak_time_target = None,
            peak_time_sd     = None,
            peak_y_target    = None,
            peak_y_sd        = None,
            total_y_target   = None,
            total_y_sd       = None,
            peak_temperature = None,     #--accepted for parity; unused
            peak_a_to_b      = None,     #--CHANGE (B): if given, adds the B component
            num_warmup       = 1000,
            num_samples      = 1000,
            num_chains       = 1,
            dense_mass       = True,     #--CHANGE (P): learn amplitude<->new_peak<->intercept correlation
            target_accept_prob = 0.95,   #--CHANGE (P): smaller steps to crawl the ridge
            max_tree_depth   = 12,       #--CHANGE (P): allow longer trajectories along the ridge
            seed             = 20200320):

        self.peak_time_target = peak_time_target
        self.peak_time_sd     = peak_time_sd
        self.peak_y_target    = peak_y_target
        self.peak_y_sd        = peak_y_sd
        self.total_y_target   = total_y_target
        self.total_y_sd       = total_y_sd
        self.peak_a_to_b      = peak_a_to_b   #--CHANGE (B)

        fit_kwargs = self._model_kwargs(peak_time_target, peak_time_sd,
                                        peak_y_target, peak_y_sd,
                                        total_y_target, total_y_sd,
                                        forecast=None)

        #--CHANGE (P): the amplitude prior now fights the rising-limb fit along a
        #--curved amplitude<->new_peak<->intercept ridge. A dense mass matrix lets
        #--NUTS learn that correlation (fixes the low n_eff / high split-r_hat on
        #--amplitude & new_peak); the higher target_accept + tree depth let it crawl
        #--the ridge instead of drifting across it.
        kernel = NUTS(self.model,
                      init_strategy=init_to_median(num_samples=100),
                      find_heuristic_step_size=True,
                      dense_mass=dense_mass,
                      target_accept_prob=target_accept_prob,
                      max_tree_depth=max_tree_depth)
        mcmc = MCMC(kernel,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                    jit_model_args=False)
        mcmc.run(jax.random.PRNGKey(seed),
                 extra_fields=("diverging", "num_steps", "accept_prob"),
                 **fit_kwargs)

        self.mcmc = mcmc
        mcmc.print_summary()
        self.posterior_samples = mcmc.get_samples()
        return self

    # ------------------------------------------------------------------
    # SVI fit (fast debugging, mirrors puca.fit_vi)
    # ------------------------------------------------------------------
    def fit_vi(self,
               peak_time_target = None,
               peak_time_sd     = None,
               peak_y_target    = None,
               peak_y_sd        = None,
               total_y_target   = None,
               total_y_sd       = None,
               peak_temperature = None,
               peak_a_to_b      = None,
               guide_type       = "lowrank",
               num_steps        = 8000,
               learning_rate    = 5e-3,
               clip_norm        = 10.0,
               num_samples      = 1000,
               rank             = 10,
               init_scale       = 0.05,
               seed             = 20200320):
        from numpyro.infer.autoguide import (AutoNormal,
                                             AutoLowRankMultivariateNormal,
                                             AutoMultivariateNormal,
                                             AutoDelta)

        self.peak_time_target = peak_time_target
        self.peak_time_sd     = peak_time_sd
        self.peak_y_target    = peak_y_target
        self.peak_y_sd        = peak_y_sd
        self.total_y_target   = total_y_target
        self.total_y_sd       = total_y_sd
        self.peak_a_to_b      = peak_a_to_b   #--CHANGE (B)

        fit_kwargs = self._model_kwargs(peak_time_target, peak_time_sd,
                                        peak_y_target, peak_y_sd,
                                        total_y_target, total_y_sd,
                                        forecast=None)

        init_fn = init_to_median(num_samples=100)
        if   guide_type == "normal":
            guide = AutoNormal(self.model, init_loc_fn=init_fn, init_scale=init_scale)
        elif guide_type == "lowrank":
            guide = AutoLowRankMultivariateNormal(self.model, rank=rank, init_loc_fn=init_fn, init_scale=init_scale)
        elif guide_type == "mvn":
            guide = AutoMultivariateNormal(self.model, init_loc_fn=init_fn, init_scale=init_scale)
        elif guide_type == "delta":
            guide = AutoDelta(self.model, init_loc_fn=init_fn)
        else:
            raise ValueError(f"unknown guide_type {guide_type!r}; use normal|lowrank|mvn|delta")

        optimizer = numpyro.optim.ClippedAdam(step_size=learning_rate, clip_norm=clip_norm)
        svi       = SVI(self.model, guide, optimizer, loss=Trace_ELBO())

        rng = jax.random.PRNGKey(seed)
        svi_result = svi.run(rng, num_steps, **fit_kwargs)

        self.guide      = guide
        self.svi_result = svi_result
        self.svi_losses = svi_result.losses
        print(f"[puca_shapev2.fit_vi] guide={guide_type} steps={num_steps} "
              f"final ELBO loss={float(svi_result.losses[-1]):.2f}")

        rng, rng_draw = jax.random.split(rng)
        post_latent = guide.sample_posterior(rng_draw, svi_result.params,
                                             sample_shape=(num_samples,), **fit_kwargs)

        rng, rng_det = jax.random.split(rng)
        predictive = Predictive(self.model, posterior_samples=post_latent)
        det = predictive(rng_det, **fit_kwargs)

        self.posterior_samples = {**post_latent, **det}
        return self

    # ------------------------------------------------------------------
    # Forecast: re-run the model forward on the posterior draws
    # ------------------------------------------------------------------
    def forecast(self, forecast_from_derivative=None):
        """
        Re-run the model forward on posterior draws.

        V2 always integrates the fitted weekly change forward from the last
        observed level (there is no level-based branch). forecast_from_derivative
        is accepted for interface parity but ignored.
        """
        if forecast_from_derivative is not None:
            self.forecast_from_derivative = forecast_from_derivative

        fc_kwargs = self._model_kwargs(self.peak_time_target, self.peak_time_sd,
                                       self.peak_y_target, self.peak_y_sd,
                                       self.total_y_target, self.total_y_sd,
                                       forecast=True,
                                       forecast_from_derivative=forecast_from_derivative)

        extra_sites = ["y_pred", "trend_fc", "proc_error", "chosen_shape", "shape_weights"]
        return_sites = list(dict.fromkeys(list(self.posterior_samples.keys()) + extra_sites))
        predictive = Predictive(self.model,
                                posterior_samples=self.posterior_samples,
                                return_sites=return_sites)
        pred = predictive(jax.random.PRNGKey(100915), **fc_kwargs)

        self.pred_samples = pred
        forecasts = np.asarray(pred["y_pred"]).squeeze()   #--(draws, T)
        self.forecasts = forecasts
        return forecasts


if __name__ == "__main__":
    pass
