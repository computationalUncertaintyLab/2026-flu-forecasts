import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

numpyro.enable_validation(True)

from numpyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer.reparam import LocScaleReparam

# Substeps per observation interval (e.g. week). dt = 1/n_substeps; total time still T.
class puca( object ):
    
    euler_substeps = 1

    #--change (smoothing): when False, past-season Y is used raw instead of
    #--Gaussian-smoothed. Smoothing let the trend fit past seasons near-perfectly,
    #--driving observation noise -> 0. Set back to True to revert.
    smooth_past_data = False

    #--change (tau-anchor): temperature for the differentiable soft-argmax that
    #--locates each template's peak on the tau grid. Smaller = sharper (closer to
    #--the true argmax) but spikier gradients. See _template_peak_tau / the warps.
    tau_peak_temperature = 0.05

    #--change (changepoint): gate the A epidemic with a smooth logistic onset so
    #--the curve holds at the baseline (int_s) until ~(peak - onset_lead), then
    #--climbs. A pure time-warp only RELOCATES a fixed-shape bump; it cannot make
    #--the toe longer or the takeoff steeper because SIR ties those to I0/R0. The
    #--gate decouples flat-toe length (onset_lead) and takeoff sharpness
    #--(onset_width) from the SIR growth rate. It is a strict generalization: with
    #--a large onset_lead the gate is ~1 over the whole bump and the model reduces
    #--to the ungated SIR. Set False to revert.
    use_changepoint = True

    def __init__(self
                 , y                 = None
                 , Y                 = None
                 , X                 = None
                 ,anchor             = None):

        self.X__input          = X
        self.y__input          = y
        self.Y__input          = Y
        self.anchor            = anchor  
        
        self.organize_data()

    def organize_data(self):
        def smooth_gaussian_anchored_nan_safe(x, sigma=1.0, keep_nan_positions=True):
            x = np.asarray(x, float)
            is_1d = (x.ndim == 1)
            if is_1d:
                x = x.reshape(-1, 1)

            radius = int(3 * sigma)
            t = np.arange(-radius, radius + 1)
            kernel = np.exp(-0.5 * (t / sigma) ** 2)
            kernel /= kernel.sum()

            mask = np.isfinite(x).astype(float)
            x0   = np.where(np.isfinite(x), x, 0.0)

            # pad both signal and mask the same way
            x_pad = np.pad(x0,   ((radius, radius), (0, 0)), mode="reflect")
            m_pad = np.pad(mask, ((radius, radius), (0, 0)), mode="reflect")

            y = np.zeros_like(x0)
            for i in range(x.shape[1]):
                num_full = np.convolve(x_pad[:, i], kernel, mode="same")
                den_full = np.convolve(m_pad[:, i], kernel, mode="same")

                num = num_full[radius:-radius]
                den = den_full[radius:-radius]

                yi = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-12)
                y[:, i] = yi

            # anchor endpoints only if they exist (finite)
            for i in range(x.shape[1]):
                if np.isfinite(x[0, i]):  y[0, i]  = x[0, i]
                if np.isfinite(x[-1, i]): y[-1, i] = x[-1, i]

            if keep_nan_positions:
                y = np.where(np.isfinite(x), y, np.nan)

            return y.ravel() if is_1d else y

        def smooth_y_data(y,Y):
            all_y             = np.array([])
            smooth_ys         = [] 
            y_means, y_scales = [] , []

            for n,(past,current) in enumerate(zip(Y.T,y)):
                means  = np.mean(past,axis=0)
                scales =  np.std(past,axis=0)

                y_means.append(means)
                y_scales.append(scales)

                #--change (smoothing): gate smoothing on the class flag so it is
                #--easy to revert. When off, use the raw past-season series.
                if puca.smooth_past_data:
                    smooth_y   =  smooth_gaussian_anchored_nan_safe(x=past,sigma=1)
                else:
                    smooth_y   =  np.asarray(past, float)
                smooth_ys.append(smooth_y)

                if n==0:
                    all_y = np.array([smooth_y])
                else:
                    all_y = np.vstack([all_y,smooth_y])
            return all_y.T

        def center_scale(y):
            global_mu  = np.nanmean( np.nanmean(y,0))
            global_std = np.nanmean( np.nanstd(y,0))
            y = (y - global_mu) / global_std

            return y, (global_mu, global_std)

        #--ROUTINE STARTED
        y           =  self.y__input[0] #<-- for now we assume one target only
        Y           =  self.Y__input[0]
        X           =  None  #self.X__input

        #--This block standardizes Y data to z-scores, collects the mean and sd, and smooths past Ys.
        smoothed_ys                                             = smooth_y_data( y,Y )

        #--remove the current season from this computation
        centered_smoothed_ys, (self.global_mu, self.global_std) = center_scale( smoothed_ys )

        #register( centered_smoothed_ys )

        #--record the last observations from the current season y
        try:
            tobs =  int( min(np.argwhere(np.isnan(y))) )
        except:
            tobs = None
        self.tobs = tobs

        #--record number of copies of y from past seasons
        copies      = [y.shape[-1] for y in Y]
        self.copies = copies
            
        #--STORE items
        self.T        = Y.shape[0]  #<--assumption here is Y must be same row for all items in list

        self.y      = y                     #<--Target
        self.Y      = centered_smoothed_ys  #<--remove the current season
        self.X      = X                         #<--covariate information 

        return y, Y, X

    @staticmethod
    def model( y          = None
              ,X          = None
              ,anchor     = None 
              ,global_mu  = None
              ,global_std = None
              ,forecast   = False
              ,peak_time_target = None
              ,peak_time_sd     = None
              ,peak_y_target    = None
              ,peak_y_sd        = None
              ,total_y_target   = None
              ,total_y_sd       = None
              ,peak_temperature = 0.1
              ,B                = None
              ,peak_a_to_b      = None):
        eps = 10**-6

        T, S  = X.shape
        S     = S+1 #<--adding in the y season

        #W     = B.shape[-1]
        L    = 1

        
        def sir_euler_incidence(
            S0,
            I0,
            R0,
            beta,
            gamma,
            T,
            n_substeps,
        ):
            """
            Euler integration of an SIR model with cumulative infections C.

            Uses n_substeps per observation interval (dt = 1/n_substeps) so the
            same calendar length T is covered with finer steps. Cumulative C is
            then read off at the original integer times t = 0,1,...,T (indices
            0, n_substeps, 2*n_substeps, ...); incidence per interval is the
            difference (equivalent to summing sub-step dC, since C is integrated
            linearly in the Euler steps).

            Returns
            -------
            inc : length T
                New infections over each original time interval [t, t+1).
            """
            N = S0 + I0 + R0
            C0 = 0.0
            n_steps = T * n_substeps
            dt = 1.0 / float(n_substeps)
            beta = jnp.repeat(beta, n_substeps)

            def step_fn(state, array):
                S, I, R, C = state
                beta_t     = array

                #--Numerical guards so the explicit-Euler SIR can't overflow to
                #--inf/nan when beta is huge (VI tail draws / wide priors give
                #--beta_s = exp(large)). Keep states in [0, N] and beta finite,
                #--and cap new infections/recoveries at the available S / I so S
                #--can't go negative and oscillate to infinity. These clamps only
                #--bind in pathological regions the valid posterior never visits,
                #--so NUTS behavior in-support is unchanged.
                beta_t = jnp.clip(beta_t, 0.0, 1e4)
                S      = jnp.clip(S, 0.0, N)
                I      = jnp.clip(I, 0.0, N)

                new_inf = beta_t * S * I / N
                new_rec = gamma * I

                new_inf = jnp.clip(new_inf, 0.0, S / dt)
                new_rec = jnp.clip(new_rec, 0.0, I / dt)

                S_next = S - dt * new_inf
                I_next = I + dt * (new_inf - new_rec)
                R_next = R + dt * new_rec
                C_next = C + dt * new_inf

                return (S_next, I_next, R_next, C_next), C_next

            init_state = (S0, I0, R0, C0)
            _, C_path = jax.lax.scan(step_fn, init_state, xs=beta)

            C_path        = jnp.concatenate([jnp.array([C0]), C_path])
            inc           = jnp.diff(C_path)
            return inc

        #T,S = y.shape

        # ------------------------------------------------------------
        # Hierarchical repo: repo_s[s] ~ lognormal(log(repo_global), repo_scale_local)
        # ------------------------------------------------------------
        #--Tightened priors (std 1.0 -> 0.5) on the SIR seed/susceptible logits.
        #--The wide std let logit_I0 wander to ~-2.87 (a ~5% seed) which, paired
        #--with the R0 ridge, put the SIR in the subcritical/decay regime (no
        #--epidemic bump). std=0.5 keeps I0 small (~[0.007, 0.047] at 2 sigma)
        #--and S0 high, closing off that degenerate corner.
        logit_S0 = numpyro.sample("logit_S0", dist.Normal( 3. , 0.5))
        logit_I0 = numpyro.sample("logit_I0", dist.Normal(-5.0, 0.5))

        I0       = jax.nn.sigmoid(logit_I0)
        S0       = jax.nn.sigmoid(logit_S0) - I0

        #--For Influenza A---------------------------------------------------------------------------------------
        repo_global      = numpyro.sample("repo_global", dist.Gamma(2, 1.0))

        #--PC prior on the between-season R0 scale: P(repo_scale_local > U) = alpha.
        #--Reverted U 1.0 -> 0.25: U=1 unleashed the SIR R0/I0 non-identifiability
        #--ridge (chain slid into subcritical R0~1 with a large seed -> incidence
        #--decays instead of peaking), which broke the SIR shape and blew up
        #--divergences. Keep the tight prior that pulls seasons toward repo_global.

        U                = 0.25
        alpha            = 0.05
        lamb             = -jnp.log(alpha)/U
        repo_scale_local = numpyro.sample("repo_scale_local"     , dist.Exponential(lamb))
        
        #--centered form; LocScaleReparam (with lifted, learnable centering)
        #--adapts between centered/non-centered per season.
        #with numpyro.plate("season_repo", S):
        #   log_repo_s = numpyro.sample("log_repo_s", dist.Normal(jnp.log(repo_global), repo_scale_local))

        log_repo_s = numpyro.sample("log_repo_s", dist.Normal(jnp.log(repo_global), repo_scale_local))
        repo_s     = jnp.repeat(jnp.exp(log_repo_s),S)
        
        numpyro.deterministic("repo_s", repo_s)

        gamma  = numpyro.sample("gamma", dist.Gamma(6, 6))

        beta_base_s   = repo_s * gamma

        #--HIERARCHICAL PER-SEASON SPLINE COEFFICIENTS (replaces the per-season beta
        #--random walk). The RW (S*(T-1)=~168 latents) was redundant with the shared
        #--spline, made beta non-SIR-flexible, and hurt geometry/speed. Instead:
        #--  mu_coef  : shared coefficient vector -> the COMMON log-beta shape
        #--  coef_s   : each season's coefficients, shrunk toward mu_coef by tau_coef
        #--This is smooth by construction (spline basis), costs only K*(S+1)+2 latents,
        #--and the shrinkage lets data-poor seasons (esp. the partially-observed target)
        #--revert to the shared shape instead of wandering.
        Bmat  = jnp.asarray(B)          #--(T, K) spline basis
        K     = Bmat.shape[-1]

        #--Shared shape: non-centered coefficient vector, scale ~ PC prior.
        U           = 0.10; alpha = 0.05; lamb_scale = -jnp.log(alpha)/U
        scale       = numpyro.sample("scale", dist.Exponential(lamb_scale))
        z_mu_coef   = numpyro.sample("z_mu_coef", dist.Normal(0.0, 1.0).expand([K]))
        mu_coef     = numpyro.deterministic("mu_coef", scale * z_mu_coef)          #--(K,)

        #--Between-season shape variation: tau_coef ~ PC prior (tight -> strong
        #--shrinkage toward the shared shape). Non-centered season deviations.
        U           = 0.10; alpha = 0.05; lamb_tau = -jnp.log(alpha)/U
        tau_coef    = numpyro.sample("tau_coef", dist.Exponential(lamb_tau))
        z_coef_s    = numpyro.sample("z_coef_s", dist.Normal(0.0, 1.0).expand([S, K]).to_event(2))
        coef_s      = numpyro.deterministic("coefficients", mu_coef[None, :] + tau_coef * z_coef_s)  #--(S,K)

        spline_beta_s = coef_s @ Bmat.T                                            #--(S,T)
        spline_beta_s = spline_beta_s - jnp.mean(spline_beta_s, axis=1, keepdims=True)

        #--cap the log-beta exponent so a tail draw of scale/coefficients can't
        #--make beta_s = exp(huge) = inf before it reaches the integrator. a_max
        #--= 9 -> beta_s <= ~8.1e3; valid draws sit near log(beta_base) ~ [-1, 2]
        #--so this only clips genuinely explosive (off-support) samples.
        log_beta_s    = jnp.clip( jnp.log( beta_base_s[:,None] + eps ) + spline_beta_s, a_max=9.0 )
        beta_s        = numpyro.deterministic("beta_s", jnp.exp( log_beta_s ) )

        inca = jax.vmap(
            lambda b: sir_euler_incidence(
                S0=S0,
                I0=I0,
                R0=0.0,
                beta=b,
                gamma=gamma,
                T=T,
                n_substeps=puca.euler_substeps), in_axes=0)(beta_s)
        
        #--change (1): normalize each season's incidence to unit peak so the SIR
        #--contributes only shape and a_s carries the amplitude. This removes the
        #--multiplicative ridge between the SIR magnitude and a_s. (Reverted: the
        #--/max(inca) division couples the whole trajectory through one denominator,
        #--which empirically RAISED divergences (0->5) and halved ESS/sec with no
        #--tree-depth benefit. Re-enable by uncommenting the line below.)
        #inca = inca / (jnp.max(inca, axis=1, keepdims=True) + eps)
        numpyro.deterministic("inca", inca)

        peaks       = jnp.nanargmax(X, axis=0)
        delta_std  = numpyro.sample("delta_std"    , dist.HalfNormal(1.0))
        new_peak   = numpyro.sample("new_peak", dist.Normal(jnp.mean(peaks), delta_std))
        all_peaks =  numpyro.deterministic("all_peaks", jnp.append(peaks, new_peak))

        U         = 0.10
        alpha     = 0.05
        lamb      = -jnp.log(alpha)/U

        calendar_time = jnp.arange(0, T)
        original_taus = (2 * calendar_time + T + 0) / (2 * T)

        #--change (tau-anchor): find where each season's template actually peaks on
        #--the tau grid (differentiable soft-argmax). Normalize to unit peak first
        #--so tau_peak_temperature is on a consistent scale regardless of amplitude.
        def _template_peak_tau(inc, taus, temperature):
            inc_n = inc / (jnp.max(inc, axis=1, keepdims=True) + eps)
            w     = jax.nn.softmax(inc_n / temperature, axis=1)   #--(S,T)
            return jnp.sum(w * taus[None, :], axis=1)             #--(S,)

        #--For A: anchor the warp to the template's real peak-tau instead of a
        #--hard-coded 1, so main_trenda peaks exactly at all_peaks regardless of
        #--where the SIR incidence crests on the euler grid (it crests at tau~0.64,
        #--not 1, which was shifting realized peaks ~13-15 weeks early).
        tau_peak_a = _template_peak_tau(inca, original_taus, puca.tau_peak_temperature)
        numpyro.deterministic("tau_peak_a", tau_peak_a)
        
        ha =  tau_peak_a[None, :] + (calendar_time[:, None]-all_peaks[None,:])/T
        #ha =   1 + (calendar_time[:, None]-all_peaks[None,:])/T
        ha = numpyro.deterministic("ha", ha)

        main_trenda = jax.vmap( lambda hh, inc_row: jnp.interp(hh, original_taus, inc_row),in_axes=(1, 0),)(ha, inca)
        main_trenda = main_trenda.T
        numpyro.deterministic("main_trenda", main_trenda)

        #--change (changepoint): smooth logistic onset gate for the A component.
        #--onset_s = peak - onset_lead (per season, using the peaks we already
        #--anchor), so onset_lead is the flat-toe length shared across seasons and
        #--onset_width is the takeoff sharpness. gate ~0 before onset, ~1 after, so
        #--a_s*main_trenda*gate stays flat at baseline then climbs steeply. Past
        #--seasons (full curves) identify onset_lead/onset_width and transfer them
        #--to the partially-observed target. Strict generalization: large lead ->
        #--gate ~1 everywhere -> reduces to the ungated bump. Toggle: use_changepoint.
        if puca.use_changepoint:
            onset_lead  = numpyro.sample("onset_lead",  dist.Gamma(6.0, 1.0))   #--mean ~6 wk toe
            onset_width = numpyro.sample("onset_width", dist.Gamma(3.0, 2.0))   #--mean ~1.5 wk takeoff
            onset_a     = all_peaks - onset_lead                                #--(S,)
            gate_a      = numpyro.deterministic(
                "gate_a",
                jax.nn.sigmoid((calendar_time[:, None] - onset_a[None, :]) / (onset_width + eps)))
        else:
            gate_a      = jnp.ones((T, S))

        U         = 0.10
        alpha     = 0.05
        lamb      = -jnp.log(alpha)/U

        mu_int      = numpyro.sample("mu_int", dist.Normal(0.0, 0.1))
        tau_int  = numpyro.sample("tau_int", dist.Exponential( lamb ) )
        
        U         = 0.10
        alpha     = 0.05
        lamb      = -jnp.log(alpha)/U

        tau_log_a  = numpyro.sample("tau_log_a", dist.Exponential( lamb ) )
        mu_log_a   = numpyro.sample("mu_log_a" , dist.Normal(0.0, 1.0))
        with numpyro.plate("season_ab", S):
            log_a_s = numpyro.sample("log_a_s"   , dist.Normal(mu_log_a, tau_log_a))
            a_s     = numpyro.deterministic("a_s", jnp.exp(log_a_s))

            #--intercept
            int_s     = numpyro.sample("int_s", dist.Normal(mu_int, tau_int))
            
        #--For Influenza B---------------------------------------------------------------------------------------
        if peak_a_to_b is not None:
            #==CHANGE (B2A) begin: wire B to A's template ========================
            #--B has no data (no observed double peak) to identify its own SIR, so
            #--its own repo/beta/incb were unidentified (a divergence source) and
            #--incb sat at its natural early peak (tau~0.7). Combined with the warp
            #--assuming tau=1, that pulled B's realized peak ~13 wks earlier and
            #--cancelled the regression's +12 wk gap -> B landed on top of A.
            #--Reusing inca (already relocated to tau~1 by A's spline+data) makes B
            #--a clean SIR bump anchored at all_peaks_b, so peak_b > peak_a holds.
            #--To revert: delete this block and uncomment the B-own-SIR block below.
            #incb = numpyro.deterministic("incb", inca)
            #==CHANGE (B2A) end =================================================

            #==CHANGE (B2A): original B-own-SIR block (uncomment to restore) =====
            repo_global      = numpyro.sample("repo_global_b", dist.Gamma(2, 1.0))
            #--PC prior mirrored from the A block: P(repo_scale_local_b > U) = alpha,
            #--reverted U 1.0 -> 0.25 alongside the A block (see note above).
            U                = 0.25
            alpha            = 0.05
            lamb             = -jnp.log(alpha)/U
            repo_scale_local = numpyro.sample("repo_scale_local_b"     , dist.Exponential(lamb))
            with numpyro.plate("season_repo", S):
               log_repo_s_b = numpyro.sample("log_repo_s_b", dist.Normal(jnp.log(repo_global), repo_scale_local))
            repo_s = jnp.exp(log_repo_s_b)
            numpyro.deterministic("repo_s_b", repo_s)
            beta_base_s   = repo_s * gamma
            beta_s        = beta_base_s[:,None]+jnp.zeros((S,T))
            incb = jax.vmap(
                lambda b: sir_euler_incidence(
                    S0    = S0,
                    I0    = I0,
                    R0    = 0.0,
                    beta  = b,
                    gamma = gamma,
                    T     = T,
                    n_substeps=puca.euler_substeps), in_axes=0)(beta_s)
            #--change (1): normalize each season's incidence to unit peak (see inca).
            incb = incb / (jnp.max(incb, axis=1, keepdims=True) + eps)
            numpyro.deterministic("incb", incb)
            #====================================================================

            U         = 0.10
            alpha     = 0.05
            lamb      = -jnp.log(alpha)/U

            peak_b0,peak_b1 = float(peak_a_to_b["b0"].values), float(peak_a_to_b["b1"].values)
            all_peaks_b     = peak_b0 + all_peaks*peak_b1

            #--change (tau-anchor): same fix as A, anchor B to its template peak-tau
            #--so main_trendb peaks exactly at all_peaks_b.
            tau_peak_b = _template_peak_tau(incb, original_taus, puca.tau_peak_temperature)
            numpyro.deterministic("tau_peak_b", tau_peak_b)
            
            hb =  tau_peak_b[None, :] + (calendar_time[:, None]-all_peaks_b[None,:])/T
            hb = numpyro.deterministic("hb", hb)

            main_trendb = jax.vmap( lambda hh, inc_row: jnp.interp(hh, original_taus, inc_row),in_axes=(1, 0),)(hb, incb)
            main_trendb = main_trendb.T
            numpyro.deterministic("main_trendb", main_trendb)

            U          = 0.10
            alpha      = 0.05
            lamb       = -jnp.log(alpha)/U

            tau_log_b  = numpyro.sample("tau_log_b", dist.Exponential( lamb ) )
            mu_log_b   = numpyro.sample("mu_log_b" , dist.Normal(0.0, 1.0))

            with numpyro.plate("season_ab", S):
                #--For B
                log_b_s = numpyro.sample("log_b_s", dist.Normal(mu_log_b, tau_log_b))
                b_s     = numpyro.deterministic("b_s", jnp.exp(log_b_s))

            #--Record the realized peak of the B component per season so it can be
            #--observed. peak_time_b = week index of the component maximum;
            #--peak_y_b = component height on the raw case scale. argmax/max are
            #--recording-only (not in any gradient path), so non-smoothness is fine.
            b_component = b_s[None, :] * main_trendb            #<--TxS
            numpyro.deterministic("peak_time_b", jnp.argmax(b_component, axis=0))
            numpyro.deterministic("peak_y_b",    jnp.max(b_component, axis=0) * global_std)

            mu    = numpyro.deterministic("mu", int_s[None, :] + (a_s[None, :] * main_trenda * gate_a) + (b_s[None, :] * main_trendb) )
        else:
            mu    = numpyro.deterministic("mu", int_s[None, :] + (a_s[None, :] * main_trenda * gate_a) )

        #--Record the realized peak of the A component per season so it can be
        #--observed (same convention as B above): peak_time_a = week index of the
        #--component maximum, peak_y_a = component height on the raw case scale.
        #--Include the onset gate so the recorded peak matches the gated curve.
        a_component = a_s[None, :] * main_trenda * gate_a       #<--TxS
        numpyro.deterministic("peak_time_a", jnp.argmax(a_component, axis=0))
        numpyro.deterministic("peak_y_a",    jnp.max(a_component, axis=0) * global_std)

        trend = mu

        # Hierarchical observation noise (per season): symmetric on the log scale.
        # log_sigma_s ~ Normal(mu_log_sigma, tau_log_sigma) lets a season be either
        # quieter OR noisier than the global level (the old HalfNormal added a
        # strictly-positive term, so seasons could only be noisier). tau_log_sigma
        # is now a single global between-season scale rather than S fixed values.
        #--Floor the noise: sigma = sigma_floor + exp(log_sigma_s). On z-scored data
        #--the smoothed past seasons can otherwise be fit near-perfectly, driving
        #--sigma -> 0, which makes the likelihood curvature (~1/sigma^2) explode and
        #--produces divergences. The floor keeps observation noise strictly positive.
        sigma_floor   = 0.05

        #==CHANGE (GLOBAL-SIGMA) begin: single global observation noise ==========
        #--The per-season noise hierarchy (mu_log_sigma / tau_log_sigma / log_sigma_s)
        #--was the worst-mixing block: with only S~4 seasons, tau_log_sigma is barely
        #--identified (a funnel), giving r_hat~1.12 across all log_sigma_s. Collapse
        #--to one global sigma. Keeps the floor; sigma is now a scalar broadcast over
        #--all seasons/times.
        #--To revert: comment this block and uncomment the hierarchy below; also add
        #--"log_sigma_s" back into _REPARAM_SITES and restore sigma[:-1]/sigma[-1]
        #--indexing in the llx/lly/forecast likelihoods.
        
        log_sigma = numpyro.sample("log_sigma", dist.Normal( jnp.log(1./2), jnp.sqrt(2)/2 ))
        sigma     = numpyro.deterministic("sigma", sigma_floor + jnp.exp(log_sigma))
        #==CHANGE (GLOBAL-SIGMA) end ===========================================

        #==CHANGE (GLOBAL-SIGMA): original per-season hierarchy (uncomment to restore)
        #mu_log_sigma  = numpyro.sample("mu_log_sigma", dist.Normal( jnp.log(1./2), jnp.sqrt(2)/2 ) )
        #tau_log_sigma = numpyro.sample("tau_log_sigma", dist.HalfNormal( 0.1 ))
        #with numpyro.plate("season", S):
        #    log_sigma_s = numpyro.sample("log_sigma_s", dist.Normal(mu_log_sigma, tau_log_sigma))
        #sigma = numpyro.deterministic("sigma", sigma_floor + jnp.exp(log_sigma_s))
        #======================================================================

        #--X likelihood
        numpyro.sample( "llx", dist.Normal(trend[:,:-1], sigma), obs = X.reshape(T,S-1) )

        #--y likelihood
        with numpyro.handlers.mask(mask=jnp.isfinite(y.reshape(T,))):
            numpyro.sample( "lly", dist.Normal(trend[:,-1].reshape(T,), sigma), obs = y.reshape(T,) )

        #--SOFT PENALTIES ON THE TARGET CURVE
        #-------------------------------------------
        #--All penalties are applied to the target season on the original case scale.
        numpyro.deterministic("trends",trend* global_std + global_mu)
        
        target_raw = trend[:,-1] * global_std + global_mu

        #--Scale-invariant soft-argmax weights. The old form softmax(target_raw /
        #--peak_temperature) ran on the RAW case scale (hundreds-thousands), so any
        #--peak_temperature ~ O(1-50) saturated to a one-hot argmax => the peak_time
        #--penalty carried essentially zero gradient (and its right value depended on
        #--each location's magnitude). Normalize the curve to unit dynamic range
        #--first: the logit is 0 at the peak and -1/peak_temperature at the trough,
        #--so peak_temperature is now dimensionless and identical across locations.
        #--Smaller peak_temperature = sharper (closer to true argmax, less centroid
        #--bias); larger = smoother gradient. ~0.05-0.1 is a good balance.
        peak_val   = jnp.max(target_raw)
        range_raw  = peak_val - jnp.min(target_raw) + eps
        w_peak     = jax.nn.softmax((target_raw - peak_val) / (peak_temperature * range_raw))

        soft_peak_time = numpyro.deterministic(
            "soft_peak_time",
            jnp.sum(calendar_time * w_peak)
        )
        
        peak_time = jnp.argmax(target_raw)

        soft_peak_y = numpyro.deterministic(
            "soft_peak_y",
            jnp.sum(target_raw * w_peak)
        )

        total_y = numpyro.deterministic(
            "total_y",
            jnp.sum(target_raw)
        )

        if peak_time_target is not None:
            numpyro.sample(
                "peak_time_penalty",
                dist.Normal(peak_time_target, peak_time_sd),
                obs = soft_peak_time
            )

        if peak_y_target is not None:
            numpyro.sample(
                "peak_y_penalty",
                dist.Normal(peak_y_target, peak_y_sd),
                obs = soft_peak_y
            )

        if total_y_target is not None:
            numpyro.sample(
                "total_y_penalty",
                dist.Normal(total_y_target, total_y_sd),
                obs = total_y
            )
        #-------------------------------------------
        #-------------------------------------------
        #-------------------------------------------
        #-------------------------------------------
        #-------------------------------------------

        if forecast:
            forecast = numpyro.sample("forecast", dist.Normal(trend[:,-1], sigma))
            numpyro.deterministic("y_pred", forecast * global_std + global_mu)

    #--Hierarchical location-scale sites written in centered form. LocScaleReparam
    #--decenters them, and lifting the per-site centering to a Uniform(0,1) latent
    #--lets NUTS *learn* how centered each should be (adaptive per season).
    #==CHANGE (B2A): "log_repo_s_b" removed since B no longer samples its own SIR.
    #--To revert: add "log_repo_s_b" back into this tuple.
    #==CHANGE (GLOBAL-SIGMA): "log_sigma_s" removed (single global sigma now).
    #--To revert: add "log_sigma_s" back into this tuple.
    _REPARAM_SITES = ("log_repo_s", "log_a_s", "int_s", "new_peak", "log_b_s")

    def _inference_model(self):
        config = {name: LocScaleReparam(centered=None) for name in self._REPARAM_SITES}
        model  = numpyro.handlers.reparam(self.model, config=config)
        #--turn each `<site>_centered` param into a latent so it is inferred under MCMC
        model  = numpyro.handlers.lift(model, prior=dist.Uniform(0.0, 1.0))
        return model

    def _inference_model_vi(self):
        #--VI-compatible variant: plain NON-centered reparam (centered=0), WITHOUT
        #--the `lift` handler. lift creates learnable `<site>_centered` *param*
        #--sites whose shapes autoguides can't match (Model/guide shape disagree at
        #--'int_s_centered'), so it only works under NUTS. centered=0 gives the same
        #--funnel-breaking non-centering, creates only `<site>_decentered` sample
        #--sites, and is fully autoguide-compatible.
        config = {name: LocScaleReparam(centered=0) for name in self._REPARAM_SITES}
        return numpyro.handlers.reparam(self.model, config=config)

    def fit(self
            , M                          = 0
            , estimated_num_components_y = None
            , peak_time_target           = None
            , peak_time_sd               = None
            , peak_y_target              = None
            , peak_y_sd                  = None
            , total_y_target             = None
            , total_y_sd                 = None
            , peak_temperature           = 0.1
            , peak_a_to_b                = None):

        y, Y, X     = self.y, self.Y, self.X
        self.M      = M

        self.peak_time_target = peak_time_target
        self.peak_time_sd     = peak_time_sd
        self.peak_y_target    = peak_y_target
        self.peak_y_sd        = peak_y_sd
        self.total_y_target   = total_y_target
        self.total_y_sd       = total_y_sd
        self.peak_temperature = peak_temperature
        self.peak_a_to_b      = peak_a_to_b

        from patsy import dmatrix
        interior_knots = np.linspace(0, 2, 10)[1:-1]
        #--Spline Basis to learn a general pattern if one exists acros seasons
        B_season = dmatrix("bs(x, knots=knots, degree=3, lower_bound=0, upper_bound=2, include_intercept=False) - 1"
                           ,{"x": np.linspace(0,2,len(y)),"knots": interior_knots,})
        self.B_season = B_season
        
        #--MCMC start
        #--Group parameters that lie on the same epidemic ridge so NUTS can
        #--navigate their joint posterior. logit_S0/logit_I0 (initial state),
        #--repo_global/gamma (growth & final size), and mu_log_a/mu_int
        #--(global amplitude & baseline, which trade off against epidemic size).
        dense_blocks = [
            ("logit_S0", "logit_I0", "repo_global", "gamma", "mu_log_a", "mu_int")#, "repo_scale_local"),
        ]

        inference_model = self._inference_model()
        self._predictive_model = self._inference_model   #--model forecast() should re-run under
        nuts_kernel = NUTS(inference_model
                           , init_strategy = init_to_median(num_samples=100)
                           , dense_mass = dense_blocks
                           ,  find_heuristic_step_size=True)     
        kernel      = nuts_kernel 
        mcmc        = MCMC(kernel
                    , num_warmup     = 1000
                    , num_samples    = 1000
                    , num_chains     = 1
                    , jit_model_args = False)

        mcmc.run(jax.random.PRNGKey(20200320)
                              ,X            = Y
                              ,y            = (y - self.global_mu) / self.global_std
                              ,anchor =  self.anchor
                              ,global_mu    = self.global_mu
                              ,global_std   = self.global_std
                              ,forecast     = None 
                              ,peak_time_target = peak_time_target
                              ,peak_time_sd     = peak_time_sd
                              ,peak_y_target    = peak_y_target
                              ,peak_y_sd        = peak_y_sd
                              ,total_y_target   = total_y_target
                              ,total_y_sd       = total_y_sd
                              ,peak_temperature = peak_temperature
                              ,B                = B_season
                              ,peak_a_to_b      = peak_a_to_b 
                              ,extra_fields = ("diverging", "num_steps", "accept_prob", "energy","adapt_state.step_size"))

        self.mcmc = mcmc
        mcmc.print_summary()
        samples = mcmc.get_samples()
        self.posterior_samples = samples

        return self

    def fit_vi(self
               , peak_time_target = None
               , peak_time_sd     = None
               , peak_y_target    = None
               , peak_y_sd        = None
               , total_y_target   = None
               , total_y_sd       = None
               , peak_temperature = 0.1
               , peak_a_to_b      = None
               , guide_type       = "lowrank"   #-- "normal" | "lowrank" | "mvn" | "delta"
               , num_steps        = 20000
               , learning_rate    = 5e-4
               , clip_norm        = 10.0
               , num_particles    = 1
               , num_samples      = 1000
               , rank             = 10
               , init_scale       = 0.01
               , seed             = 20200320):
        """
        Fast variational alternative to `fit` for DEBUGGING (MCMC takes ~30 min).

        Optimizes an autoguide over the same reparameterized/lifted
        `_inference_model` used by `fit`, then draws `num_samples` from the guide
        and populates `self.posterior_samples` (latents + deterministics) in the
        exact format `fit` produces -- so `forecast()` and all your existing
        diagnostics (peak_time_a, inca, main_trenda, ...) work unchanged.

        guide_type:
          "normal"  - mean-field (fastest, but UNDERESTIMATES uncertainty and
                      cannot represent the SIR/amplitude correlations)
          "lowrank" - low-rank multivariate normal (default; captures the main
                      correlations, still fast) -- best speed/quality tradeoff here
          "mvn"     - full multivariate normal
          "delta"   - MAP point estimate (no uncertainty; quickest sanity check)

        IMPORTANT: VI can hide the non-identifiabilities NUTS exposes. Use it to
        iterate on structure / shapes / peak placement, then confirm final
        numbers and intervals with `fit` (NUTS).
        """
        from numpyro.infer.autoguide import (AutoNormal,
                                             AutoLowRankMultivariateNormal,
                                             AutoMultivariateNormal,
                                             AutoDelta)

        y, Y, X = self.y, self.Y, self.X

        self.peak_time_target = peak_time_target
        self.peak_time_sd     = peak_time_sd
        self.peak_y_target    = peak_y_target
        self.peak_y_sd        = peak_y_sd
        self.total_y_target   = total_y_target
        self.total_y_sd       = total_y_sd
        self.peak_temperature = peak_temperature
        self.peak_a_to_b      = peak_a_to_b

        from patsy import dmatrix
        interior_knots = np.linspace(0, 2, 10)[1:-1]
        B_season = dmatrix("bs(x, knots=knots, degree=3, lower_bound=0, upper_bound=2, include_intercept=False) - 1"
                           ,{"x": np.linspace(0,2,len(y)),"knots": interior_knots,})
        self.B_season = B_season

        #--exact same model kwargs as the NUTS path (forecast=None while fitting)
        model_kwargs = dict(
            X                = Y,
            y                = (y - self.global_mu) / self.global_std,
            anchor           = self.anchor,
            global_mu        = self.global_mu,
            global_std       = self.global_std,
            forecast         = None,
            peak_time_target = peak_time_target,
            peak_time_sd     = peak_time_sd,
            peak_y_target    = peak_y_target,
            peak_y_sd        = peak_y_sd,
            total_y_target   = total_y_target,
            total_y_sd       = total_y_sd,
            peak_temperature = peak_temperature,
            B                = B_season,
            peak_a_to_b      = peak_a_to_b,
        )

        #--VI uses the non-centered (no-lift) model; record it so forecast() re-runs
        #--the posterior under the same model it was drawn from.
        inference_model        = self._inference_model_vi()
        self._predictive_model = self._inference_model_vi

        #--Small init_scale keeps the guide tight around the median early on, so it
        #--does not sample explosive tails (large scale*z -> beta_s=exp(huge) ->
        #--the Euler SIR overflows -> nan ELBO). Grow it later if you want wider
        #--posteriors once the fit is stable.
        init_fn = init_to_median(num_samples=100)
        if   guide_type == "normal":
            guide = AutoNormal(inference_model, init_loc_fn=init_fn, init_scale=init_scale)
        elif guide_type == "lowrank":
            guide = AutoLowRankMultivariateNormal(inference_model, rank=rank, init_loc_fn=init_fn, init_scale=init_scale)
        elif guide_type == "mvn":
            guide = AutoMultivariateNormal(inference_model, init_loc_fn=init_fn, init_scale=init_scale)
        elif guide_type == "delta":
            guide = AutoDelta(inference_model, init_loc_fn=init_fn)
        else:
            raise ValueError(f"unknown guide_type {guide_type!r}; use normal|lowrank|mvn|delta")

        #--ClippedAdam (gradient clipping) instead of plain Adam: this model's
        #--likelihood is stiff (1/sigma^2 with a noise floor, plus sharp soft-peak
        #--penalties), so unclipped steps drive the ELBO to nan almost immediately.
        optimizer = numpyro.optim.ClippedAdam(step_size=learning_rate, clip_norm=clip_norm)
        svi       = SVI(inference_model, guide, optimizer, loss=Trace_ELBO(num_particles=num_particles))

        rng = jax.random.PRNGKey(seed)
        svi_result = svi.run(rng, num_steps, **model_kwargs)

        self.guide      = guide
        self.svi_result = svi_result
        self.svi_params = svi_result.params
        self.svi_losses = svi_result.losses
        print(f"[fit_vi] guide={guide_type} steps={num_steps} final ELBO loss={float(svi_result.losses[-1]):.2f}")

        #--draw latent posterior samples from the fitted guide. Pass the model
        #--kwargs: with reparam the guide holds `_decentered` latents and must
        #--re-trace the model (needs X, B, ...) to map them back / fill sites.
        rng, rng_draw = jax.random.split(rng)
        post_latent = guide.sample_posterior(rng_draw, svi_result.params,
                                             sample_shape=(num_samples,), **model_kwargs)

        #--recompute deterministics by running the model on those latent draws so
        #--posterior_samples matches the (latents + deterministics) format NUTS
        #--produces, and forecast()/diagnostics work unchanged.
        rng, rng_det = jax.random.split(rng)
        predictive = Predictive(inference_model, posterior_samples=post_latent, return_sites=None)
        det        = predictive(rng_det, **model_kwargs)

        self.posterior_samples = {**post_latent, **det}
        return self

    def forecast(self):

        #--MCMC START
        #--Re-run the posterior under the SAME model it was drawn from: the
        #--lifted/reparam model for NUTS (fit), or the non-centered model for VI
        #--(fit_vi). _predictive_model is set by whichever fit was called.
        predictive_model = getattr(self, "_predictive_model", self._inference_model)
        predictive = Predictive(predictive_model(), posterior_samples = self.posterior_samples
                                , return_sites               = list(self.posterior_samples.keys()) + ["y_pred"] )
        #--MCMC END

        rng_key    = jax.random.PRNGKey(100915)
        pred_samples = predictive( rng_key
                              ,X                =  self.Y
                              ,y                = (self.y - self.global_mu) / self.global_std
                              ,anchor           =  self.anchor
                              ,global_mu        = self.global_mu
                              ,global_std       = self.global_std
                              ,forecast         = True
                              ,peak_time_target = self.peak_time_target
                              ,peak_time_sd     = self.peak_time_sd
                              ,peak_y_target    = self.peak_y_target
                              ,peak_y_sd        = self.peak_y_sd
                              ,total_y_target   = self.total_y_target
                              ,total_y_sd       = self.total_y_sd
                              ,peak_temperature = self.peak_temperature
                              ,B                = self.B_season
                              ,peak_a_to_b      = self.peak_a_to_b 
                                  )
        yhat_draws = pred_samples["y_pred"]      # (draws, T, S)

        yhat_draws = yhat_draws.squeeze()

        forecasts = yhat_draws
        
        self.pred_samples = pred_samples
        self.forecast     = forecasts
        return forecasts

if __name__ == "__main__":
    pass
