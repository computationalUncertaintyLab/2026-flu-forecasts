#mcandrew

import sys
import numpy as np
import pandas as pd

from glob import glob

from scipy.optimize import minimize
from scipy.special import softmax

from datetime import datetime, timedelta

def WIS(d, col, observed):
    """Weighted interval score for a quantile forecast.

    Parameters
    ----------
    d : pd.DataFrame
        Indexed by quantile level (`output_type_id`), with a `combined` column
        of predictive quantiles.
    observed : float
        Realized value.

    Returns
    -------
    float
        Bracher et al. WIS (FluSight / scoringutils convention).
    """
    y = float(observed)
    taus = np.asarray(d.index, dtype=float)
    qs = np.asarray(d[col], dtype=float)

    order = np.argsort(taus)
    taus = taus[order]
    qs = qs[order]

    # 1/2 |y - median|
    med_idx = np.flatnonzero(np.isclose(taus, 0.5))
    abs_err = 0.5 * np.abs(y - qs[med_idx[0]]) if len(med_idx) else 0.0

    # (α/2) * IS_α for each central interval from paired quantiles α/2 and 1-α/2
    weighted_IS = []
    for i, tau in enumerate(taus):
        if np.isclose(tau, 0.5) or tau > 0.5 - 1e-12:
            continue
        upper = np.flatnonzero(np.isclose(taus, 1.0 - tau))
        if len(upper) == 0:
            continue
        alpha = 2.0 * tau
        lo = qs[i]
        hi = qs[upper[0]]
        IS = (hi - lo)
        if y < lo:
            IS = IS + (2.0 / alpha) * (lo - y)
        elif y > hi:
            IS = IS + (2.0 / alpha) * (y - hi)
        weighted_IS.append((alpha / 2.0) * IS)

    K = len(weighted_IS)
    return float((abs_err + np.sum(weighted_IS)) / (K + 0.5))

if __name__ == "__main__":

    all_forecasts = pd.read_csv("./multi_model_ensemble_weights/component_model_forecasts.csv")

    inc_hosps     = pd.read_csv("./data/target-data/target-hospital-admissions.csv")
    inc_hosps     = inc_hosps.loc[ (inc_hosps["date"]>="2021-10-09")  ]

    def format(x):
        if x=="US":
            return x
        return "{:02d}".format(int(x))

    all_forecasts["location"] = [format(x) for x in all_forecasts.location.values]

    inc_hosps = inc_hosps.rename(columns = {"value":"observed"})
    all_forecasts = all_forecasts.merge( inc_hosps, left_on = ["target_end_date","location"], right_on = ["date","location"] )

    all_forecasts = all_forecasts.rename(columns = {"value":"predicted"})
    
    columns =  ["reference_date","horizon","target_end_date","location","output_type_id",  "predicted", "model", "observed"]
    all_forecasts = all_forecasts[columns]
    all_forecasts = all_forecasts.loc[all_forecasts.horizon>0]

    all_forecasts = all_forecasts.loc[all_forecasts.horizon<=8]
    
    def compute_WIS_for_all(subset):
            try:
                d             = pd.pivot_table( index = ["output_type_id"], columns = ["model"], values = ["predicted"], data = subset  )
                d.columns     = [y for x,y in d.columns]
                #d["combined"] = w*d["level"] + (1-w)*d["deriv"]

                observed = float(subset.iloc[0].observed)

                wis_level = WIS( d, "level", observed )
                subset["wis_level"] = wis_level

                wis_deriv = WIS( d, "deriv", observed )
                subset["wis_deriv"] = wis_deriv

                wis_kf    = WIS( d, "KF", observed )
                subset["wis_kf"]    = wis_kf
                
                return subset
            except:
                subset["wis_level"] = np.nan
                subset["wis_deriv"] = np.nan
                subset["wis_kf"]    = np.nan

                return subset
        
    all_forecasts["target_end_date"] = pd.to_datetime(all_forecasts["target_end_date"])

    all_leaders = []
    for location, state_forecasts in all_forecasts.groupby("location"):
        dates = state_forecasts.reference_date.unique()

        #--first comute all WIS scores
        WIS_scores = state_forecasts.groupby(["location","horizon","reference_date","target_end_date"]).apply(lambda x: compute_WIS_for_all(x) )
        WIS_scores = WIS_scores.reset_index(drop=True)

        WIS_scores = WIS_scores[ ["location","horizon","reference_date","target_end_date","wis_level","wis_deriv","wis_kf"] ].drop_duplicates()
        WIS_scores["target_end_date"] = pd.to_datetime(WIS_scores["target_end_date"])

        leaders = { "location":[], "reference_date":[], "leader":[] }
        for date in dates:
            # only the past four weeks of resolved targets
            date_ts = pd.Timestamp(date)
            cut_ts  = date_ts - pd.Timedelta(weeks=4)
            subset = WIS_scores.loc[
                (WIS_scores.target_end_date < date_ts) &
                (WIS_scores.target_end_date >= cut_ts)
            ]
            if len(subset)==0: #<--first week, no eval data
                continue

            def find_avg_WIS(w, state_forecasts):
                def compute_WIS_for_all(subset, w):
                    w0 =    float(w[0])
                    w1 =    float(w[1])
                    w2 =    float(w[2])
                    try:
                        d = pd.pivot_table(index=["output_type_id"], columns=["model"], values=["predicted"], data=subset)
                        d.columns     = [y for x, y in d.columns]
                        d["combined"] = w0 * d["level"] + w1 * d["deriv"] + w2* d["KF"]

                        observed = float(subset.iloc[0].observed)

                        wis = WIS(d, "combined", observed)
                        return pd.Series({"wis": wis})
                    except Exception:
                        return pd.Series({"wis": np.nan})

                scores = (
                    state_forecasts
                    .groupby(["location", "horizon", "reference_date", "target_end_date"], group_keys=False)
                    .apply(lambda x: compute_WIS_for_all(x, w), include_groups=False)
                )
                if isinstance(scores, pd.DataFrame):
                    return float(np.nanmean(scores["wis"].to_numpy()))
                return float(np.nanmean(np.asarray(scores, dtype=float)))

            subset_forecasts = state_forecasts.loc[
                (state_forecasts.target_end_date < date_ts) &
                (state_forecasts.target_end_date >= cut_ts)
            ]

            def find_best(subset_forecasts):
                x0 = np.array([ 0.33,0.33, 0.33 ] )
                constraints = {
                    'type': 'eq', 
                    'fun': lambda x: np.sum(x) - 1
                }
                results = minimize(
                    lambda x: find_avg_WIS(x, subset_forecasts)
                    , x0     = x0
                    , bounds =[(0, 1),(0,1), (0,1)]
                    , constraints = constraints
                    , tol    =0.05
                )
                weight_for_level = float(np.asarray(results.x).ravel()[0])
                weight_for_deriv = float(np.asarray(results.x).ravel()[1])
                weight_for_kf    = 1 - (weight_for_level + weight_for_deriv)

                return pd.Series({"w_level": weight_for_level, "w_deriv": weight_for_deriv, "w_kf":weight_for_kf})

            avg_per_horizon = subset_forecasts.groupby(["horizon"]).apply( find_best ).reset_index()
            
            for horizon in np.arange(1,8+1):
                if horizon not in avg_per_horizon.horizon.values:
                    break

            #--fill in the rest
            if horizon < 8:
                last_horizon = horizon-1

                #--with no horizon data just avg them
                avg_per_horizon_extra = {"horizon":[],"w_level":[],"w_deriv":[],"w_kf":[]}
                for horizon in np.arange(horizon, 8+1):
                    avg_per_horizon_extra["horizon"].append(horizon)
                    avg_per_horizon_extra["w_level"].append( avg_per_horizon.loc[ avg_per_horizon.horizon==last_horizon,"w_level"  ].iloc[0] )
                    avg_per_horizon_extra["w_deriv"].append( avg_per_horizon.loc[ avg_per_horizon.horizon==last_horizon,"w_deriv"  ].iloc[0] )
                    avg_per_horizon_extra["w_kf"].append( avg_per_horizon.loc[ avg_per_horizon.horizon==last_horizon,"w_kf"  ].iloc[0] )
                avg_per_horizon_extra = pd.DataFrame(avg_per_horizon_extra)

                avg_per_horizon = pd.concat([avg_per_horizon,avg_per_horizon_extra])

            leaders = avg_per_horizon
            leaders["reference_date"] = date
            leaders["location"]       = location
            
            all_leaders.append(leaders)
            
    all_leaders = pd.concat(all_leaders)
    all_leaders   = all_leaders.sort_values(["location","reference_date"])

    all_leaders.to_csv("./multi_model_ensemble_weights/optimal_weights_over_time.csv",index=False)

    all_forecasts = all_forecasts.merge( all_leaders, on = ["location","reference_date","horizon"] )
        
    #--Finally, pick the model from the list
    all_forecasts = pd.pivot_table(index = ["location","reference_date","target_end_date","horizon","output_type_id","w_level","w_deriv","w_kf"], columns = ["model"], values = ["predicted"], data = all_forecasts)
    all_forecasts.columns = [y for x,y in all_forecasts.columns]
    all_forecasts = all_forecasts.reset_index()
    
    all_forecasts["value"] = all_forecasts["w_level"]*all_forecasts["level"] + all_forecasts["w_deriv"]*all_forecasts["deriv"] + all_forecasts["w_kf"]*all_forecasts["KF"]

    all_forecasts = all_forecasts.drop(columns = ["w_level","w_deriv","deriv","level","KF"])

    all_forecasts.to_csv("./multi_model_ensemble_weights/adaptive_wis_ensemble_opt.csv", index=False)
