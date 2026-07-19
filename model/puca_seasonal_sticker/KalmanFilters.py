#mcandrew
import os

import numpy as np
import pandas as pd

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

from numpyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO, init_to_median


def model_loclin(y,forecast=False):

    gamma2 = numpyro.sample("gammat", dist.Gamma(2, 0.02))
    nu2    = numpyro.sample("nut"   , dist.Gamma(2, 0.02))
    R2      = numpyro.sample("R"    , dist.Gamma(2, 0.02))

    gamma2_scale = 1./gamma2
    nu2_scale    = 1./nu2
    R2_scale     = 1./R2

    mask   = jnp.isfinite(y)
    y_fill = jnp.where(mask,y,0)

    def kf_loclin(carry,array,gamma,nu,R):
        mx_,ms_,Pa,Pb,Pc,ll_old = carry
        y,mask                  = array

        #--evolve state space
        mx = mx_+ms_
        ms = ms_

        Px  = Pa+Pb+2*Pc+gamma
        Ps  = Pb+nu
        Pxs = Pc+Pb

        #--compute yt given above
        my = mx
        Py = Px + R

        #--compute zt|yt
        innov = (y-mx)*mask
        
        S     = Px + R

        mx_up  = mx  + (Px/S)*innov
        ms_up  = ms  + (Pxs/S)*innov

        Px_up  = Px  - (Px**2)/S
        Ps_up  = Ps  - (Pxs**2)/S
        Pxs_up = Pxs - (Px*Pxs)/S

        #--ll calc

        LOG2PI = jnp.log(2.0 * jnp.pi)
        invS   = 1./S
        ll     = (-0.5 * (LOG2PI + jnp.log(S) + (innov * innov) * invS))*mask

        return jnp.array([mx_up, ms_up, Px_up,Ps_up,Pxs_up, ll_old+ll]), (mx_up,Px_up,my,Py)

    m_s      = numpyro.sample("start"   , dist.Normal(0,1).expand([2]))
    Pa_Pb    = numpyro.sample("Pa_Pb_Pc", dist.HalfNormal(1).expand([2]))
    Pc       = numpyro.sample("Pc"      , dist.Uniform(-0.99,0.99))

    mx0, ms0 = m_s[0], m_s[1]
    Pxx0, Pss0, Pxs0 = Pa_Pb[0]**2, Pa_Pb[1]**2, Pc*Pa_Pb[0]*Pa_Pb[1]
    init = jnp.array([mx0, ms0, Pxx0, Pss0, Pxs0, 0.0])

    _ , (mx,Px,my,Py) = jax.lax.scan( lambda x,y: kf_loclin(x,y,gamma2_scale,nu2_scale,R2_scale)
                           , init = init
                           , unroll=8
                           , xs   = (y_fill,mask) )
    numpyro.factor( "LL", _[-1])

    mx = numpyro.deterministic("mx", mx)
    Px = numpyro.deterministic("Px", Px)

    if forecast:
       numpyro.sample("pred", dist.Normal(my,Py))

def hier_loclin(y,forecast=False):

    beta2    = numpyro.sample("beta2"   , dist.Gamma(5, 0.005))
    gamma2   = numpyro.sample("gamma2"  , dist.Gamma(2, 0.02))
    alpha2   = numpyro.sample("alpha2"  , dist.Gamma(2, 0.02))
    eps2     = numpyro.sample("eps2"    , dist.Gamma(2, 0.02))    

    beta2_scale  = 1./beta2
    gamma2_scale = 1./gamma2
    alpha2_scale = 1./alpha2
    eps2_scale   = 1./eps2

    mask   = jnp.isfinite(y)
    y_fill = jnp.where(mask,y,0)

    def kf(carry,array,beta2,gamma2,alpha2,eps2):
        mh_,ms_,mx_, Pa,Pb,Pc,Pd,Pe,Pf,ll_old = carry
        y,mask                                = array

        #--evolve state space
        mh = mh_
        ms = ms_
        mx = ms_+mx_

        P_hh = Pa + beta2
        P_hs = Pe
        P_hx = Pd+Pe
        
        P_ss = Pc + gamma2
        P_sx = Pc + Pf

        P_xh = Pd+Pe 
        P_xs = Pc+Pf
        P_xx = Pb + 2*Pf + Pc + alpha2

        #--compute yt given above
        my = mh + mx
        Py = P_xx + eps2

        innov = y - my
        
        #--compute zt|yt
        m_h = mh + (P_hx/Py)*innov
        m_s = ms + (P_sx/Py)*innov
        m_x = mx + (P_xx/Py)*innov 
        
        P_a = P_hh-(P_hx**2)/Py
        P_b = P_ss-(P_sx**2)/Py
        P_c = P_xx-(P_xx**2)/Py
        P_d = P_hs-(P_hx*P_sx)/Py
        P_e = P_hx-(P_hx*P_xx)/Py
        P_f = P_sx-(P_sx*P_xx)/Py
        
        #--ll calc
        LOG2PI = jnp.log(2.0 * jnp.pi)
        S      = Py
        invS   = 1./S
        ll     = (-0.5 * (LOG2PI + jnp.log(S) + (innov * innov) * invS))*mask

        return jnp.array( [m_h,m_s,m_x, P_a, P_b, P_c, P_d, P_e, P_f, ll_old+ll]), (my,Py)

    m_h_s_x        = numpyro.sample("start"   , dist.Normal(0,1).expand([3]))
    mh0, ms0, mx0  = m_h_s_x[0], m_h_s_x[1]*0.2, m_h_s_x[2]
    
    corr   = numpyro.sample("corr", dist.LKJCholesky(3,2))
    scales = numpyro.sample("scales", dist.HalfNormal(1).expand([3]))

    L0 = jnp.diag(scales) @ corr
    P0 = L0 @ L0.T
    
    Pa0 = P0[0,0]
    Pb0 = P0[1,1]
    Pc0 = P0[2,2]
    Pd0 = P0[0,1]
    Pe0 = P0[0,2]
    Pf0 = P0[1,2]

    init = jnp.array([mh0,ms0,mx0,Pa0, Pb0, Pc0, Pd0, Pe0, Pf0, 0.0])

    _ , (my,Py) = jax.lax.scan( lambda x,y: kf(x,y,beta2_scale,gamma2_scale,alpha2_scale,eps2_scale)
                                , init = init
                                , unroll=8
                                , xs   = (y_fill,mask) )
    numpyro.factor( "LL", _[-1])

    if forecast:
       numpyro.sample("pred", dist.Normal(my,Py))


class run_season_hier( object ):
    def __init__(self,y):
        self.y = y
        
    def seasonal_hier(self,y, forecast=False):
        def loc_level(y,mask, sigma2, s2):
            def kf(carry,array,sigma2,s2):
                mx_,ms_, ll_ = carry
                y,mask       = array

                #--evolve xt | xt-1
                mu_xp = mx_
                P_xp  = ms_ + sigma2

                #--likelihood yt | xt
                mu_yp = mu_xp
                P_yp  = P_xp + s2

                llt = -0.5*( jnp.log(2*jnp.pi*P_yp) + ((y-mu_yp)**2)/P_yp )

                #--xt|yt
                K = P_xp / (P_xp + s2)
                mu_x_y = mu_xp + K * (y - mu_xp)
                P_x_y  = P_xp - K * P_xp

                mu_next = jnp.where( mask, mu_x_y, mu_xp )
                P_next  = jnp.where( mask, P_x_y ,  P_xp )

                ll = ll_ + llt*mask

                return (mu_next,P_next,ll) , (mu_next,P_next)

            x0 = (0,sigma2, 0 )
            (_,_,ll), (mxy,pxy) = jax.lax.scan( lambda carry,array: kf(carry,array, sigma2,s2) ,init = x0, xs = (y,mask) )

            return mxy,pxy,ll

        T,S = y.shape

        peaks = jnp.argmax(jnp.nan_to_num(y[:, :-1], nan=-jnp.inf), axis=0)
        times = jnp.arange(T, dtype=y.dtype)

        peak_guess = numpyro.sample("peak", dist.Normal(jnp.mean(peaks), 5.0))

        peaks = jnp.append(peaks.astype(y.dtype), peak_guess)                 # (S,)
        
        # season-specific tau at each calendar week: (T,S)
        tau_grid = 1.0 + (times[:, None] - peaks[None, :]) / T
        
        scale     = numpyro.sample("scale", dist.HalfNormal(1.0))
        z_all     = numpyro.sample("z_all", dist.Normal(0., 1.).expand([T]))
        intercept = numpyro.sample("intercept", dist.Normal(0., 1.))

        # shared trend on canonical tau in [0,2]
        f = intercept + jnp.cumsum(scale * z_all)                            # (T,) 1-D

        #--individual-trend
        alpha =  scale
        U     =  0.01
        lamb  =  -jnp.log(U) / alpha
        
        scale_indiv     = numpyro.sample("scale_indiv", dist.Exponential(lamb))
        noise_indiv     = numpyro.sample("noise_indiv", dist.Exponential(lamb))
       
        same_grid = jnp.linspace(0.0, 2.0, T)                                       # (T,) 1-D
        m = jax.vmap(lambda tau_s: jnp.interp(tau_s, same_grid, f),
                     in_axes=1, out_axes=1)(tau_grid)                               # (T,S)

        mask     = jnp.isfinite(y)
        res      = y - m
        res_fill = jnp.where(mask, res, 0.0)

        # hierarchical KF on residuals, one season at a time -> mxs/pxs (S,T)
        mxs,pxs,LLs = jax.vmap(
            lambda ys, ms: loc_level(ys, ms, scale_indiv**2, noise_indiv**2),
            in_axes=(1, 1),
        )(res_fill, mask)
        numpyro.factor("LL", jnp.sum(LLs))

        if forecast:
            # current season = last column; fixed-length path for JIT
            last_obs = jnp.sum(jnp.isfinite(y[:, -1])).astype(jnp.int32) - 1
            t = jnp.arange(T)

            z_forecast = numpyro.sample("z_forecast", dist.Normal(0., 1.).expand([T]))
            incr = jnp.where(t > last_obs, scale_indiv * z_forecast, 0.0)
            resid_rw = mxs[-1, last_obs] + jnp.cumsum(incr)

            # past: filtered residual; future: local-level RW from last filtered state
            resid = jnp.where(t <= last_obs, mxs[-1], resid_rw)
            numpyro.sample("y_pred", dist.Normal(m[:, -1] + resid, noise_indiv))

    def fit(self):
        nuts_kernel = NUTS(self.seasonal_hier
                           , init_strategy  = init_to_median(num_samples=100)
                           , max_tree_depth = 12
                           , dense_mass = [ ("scale","scale_indiv") ]
                           , find_heuristic_step_size=True)

        kernel      = nuts_kernel 
        mcmc        = MCMC(kernel
                           , num_warmup     = 2000
                           , num_samples    = 2500
                           , num_chains     = 1
                           , jit_model_args = False)

        mcmc.run(jax.random.PRNGKey(20200320), y = jnp.array(self.y))

        samples   = mcmc.get_samples()
        self.posterior_samples = samples
        return self

    def forecast(self):
        predictive = Predictive(self.seasonal_hier,
                                posterior_samples=self.posterior_samples, return_sites = ["y_pred"])
                                
        pred = predictive(jax.random.PRNGKey(100915), y=jnp.array(self.y), forecast=True)

        self.pred_samples = pred
        forecasts = np.asarray(pred["y_pred"]).squeeze()   #--(draws, T)
        self.forecasts = forecasts
        return forecasts

