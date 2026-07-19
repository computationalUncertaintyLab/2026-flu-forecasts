#mcandrew

#--THREAD CAPS (must be set BEFORE jax/numpy import).
#--We fan out Parallel(n_jobs=10) MCMC fits below. By default each JAX/XLA
#--process and each BLAS call grabs ALL cpu cores, so 10 concurrent fits
#--oversubscribe the machine (10 x nCores threads fighting over nCores) and
#--every fit crawls -- that's what produced the ~1hr ETA. Pin every worker to a
#--single thread so the 10 processes cleanly share the cores instead of thrashing.
import os
os.environ["XLA_FLAGS"]            = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"

# Enable 64-bit precision for better numerical stability
import jax
jax.config.update("jax_enable_x64", True)

import sys
sys.path.append('./model/')

from collections import Counter

import numpy as np
import pandas as pd

from pathlib import Path

from epiweeks import Week
from datetime import datetime, timedelta

from joblib import Parallel, delayed

import os
import pickle
from collections import Counter

from puca_shapev2 import puca_shapev2 as puca

import pickle

def from_time_to_season(x, yrstop):

    yr,week = x.MMWRYR, x.MMWRWK

    if yr==yrstop and week>=35:
        season = "-1"
        return season

    if week>20 and week<35:
        season="-1"
    else:
        if week>=40:
            season = "{:d}/{:d}".format( yr, yr+1)
        else:
            season = "{:d}/{:d}".format( yr-1, yr)
    return season

def add_time_data(row):
    from epiweeks import Week
    from datetime import datetime

    epiweek = Week.fromdate( datetime.strptime(row.date,"%Y-%m-%d"))
    row["MMWRYR"] = epiweek.year
    row["MMWRWK"] = epiweek.week
    row["season"] = "{:d}/{:d}".format(epiweek.year,epiweek.year+1) if epiweek.week >=35 else "{:d}/{:d}".format(epiweek.year-1,epiweek.year)

    return row

def interpolate_nans(array):
    """
    Linearly interpolates NaN values in a 1D NumPy array.
    For leading/trailing NaNs, it performs forward/backward filling.
    """
    nans = np.isnan(array)
    # Create an array of indices for the original array
    x = np.arange(len(array))
    
    # Use np.interp to fill NaNs
    # x=x[nans]: Indices where NaNs are present (where we want to interpolate)
    # xp=x[~nans]: Indices where non-NaN values are present (known points)
    # fp=array[~nans]: Values at the non-NaN indices (known values)
    array[nans] = np.interp(x=x[nans], xp=x[~nans], fp=array[~nans])
    
    return array

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--stack', type=str) 

    args = parser.parse_args()
    
    THIS_SEASON = "2026/2027"
    
    #--data set of populations (contains all FIPS)
    pops                = pd.read_csv("./data/locations.csv")
    
    #--incident hospitalizations dataset
    inc_hosps           = pd.read_csv("./data/target-data/target-hospital-admissions.csv")

    #--incase we want to incldue this 
    # pct_hosps_reporting = pd.read_csv("./analysis_data/pct_hospital_reporting.csv")
    # pct_hosps_reporting["season"] = pct_hosps_reporting.apply(lambda row:from_time_to_season(row,2026), 1)
    # pct_hosps_reporting = pct_hosps_reporting.loc[pct_hosps_reporting.season!='-1']
    
    #--subset by only information after 09-01
    inc_hosps           = inc_hosps.loc[ (inc_hosps["date"]>="2021-10-09")  ]

    #--ILI data
    ili_data            = pd.read_csv("./analysis_data/ili_data_all_states_2021_present__formatted.csv")
    ili_data["week"]    = [ int(str(x)[-2:]) for x in ili_data.epiweek]

    #--lab data
    lab_data            = pd.read_csv("./analysis_data/clinical_and_public_lab_data__formatted.csv")

    ili_augmented       = lab_data.merge(ili_data, on = ["state","epiweek","year","week"] )
    ili_augmented["region"] = [ "US" if x == 'National' else x for x in ili_augmented.region.values]
    
    ili_augmented = ili_augmented.merge( inc_hosps[["location","location_name"]].drop_duplicates(), left_on =["region"], right_on=["location_name"] )

    time_data = inc_hosps[["date"]].drop_duplicates()
    time_data = time_data.apply( add_time_data,1 )

    inc_hosps = inc_hosps.merge(time_data           , left_on = ["date"], right_on = ["date"] )

    ili_augmented  = ili_augmented.drop(columns = ["season"])
    ili_augmented  = ili_augmented.merge(time_data  , left_on=["year","week"], right_on=["MMWRYR","MMWRWK"] )


    def format(x):
        if x=="US":
            return x
        return "{:02d}".format(int(x))
    
    def forecast( location, season, thisweek, subset, ili_augmented ):
        print(f"Location = {location}")
        param_data = {"location":[],"season":[],"param_type":[], "param1":[],"param2":[],"value":[]}

        #all_inc_data = subset.copy()
        
        #--subset all data to specific state
        #season = "2026/2027"

        #--SKIP if this forecast file already exists (must match the write path
        #--at the bottom of this function exactly).
        _season_slug = season.replace("/","_")
        if location=="US":
            _outpath = "./real_time/shape_plus_deriv_forecasts/forecast_US_{:s}_{:s}.csv".format(_season_slug, thisweek)
        else:
            _outpath = "./real_time/shape_plus_deriv_forecasts/forecast_{:s}_{:02d}_{:s}.csv".format(_season_slug, int(location), thisweek)
        if os.path.exists(_outpath):
            print(f"SKIP (exists): {_outpath}")
            return

        past_inc_hosps_state  = past_inc_hosps.loc[(past_inc_hosps.location==location) ]
        inc_hosps_state       = subset

        print(f"{'='*60}\n")

        import jax 
        base_key    = jax.random.PRNGKey(20200320)
        worker_key  = jax.random.fold_in(base_key, 1)

        #--build Y matrix
        past_inc_hosps_state = past_inc_hosps_state.loc[past_inc_hosps_state.season!="2021/2022"]
        cases = pd.pivot_table(   index   = "MMWRWK"
                                , columns = ["season"]#, "location"]
                                , values  = "value"
                                , data    = pd.concat([past_inc_hosps_state, inc_hosps_state],axis=0), dropna=False )


        ili_augmented = ili_augmented.loc[ili_augmented.season!="2021/2022"]
        ilia = pd.pivot_table(   index   = "MMWRWK"
                                , columns = ["season"]#, "location"]
                                , values  = "percent_a"
                                , data    = ili_augmented, dropna=False )


        ilib = pd.pivot_table(   index   = "MMWRWK"
                                , columns = ["season"]#, "location"]
                                , values  = "percent_b"
                                , data    = ili_augmented, dropna=False )

        try:
            weeks = list(np.arange(35,53+1)) + list(np.arange(1,25+1))
            cases = cases.loc[weeks]
        except:
            weeks = list(np.arange(35,52+1)) + list(np.arange(1,25+1))
            cases = cases.loc[weeks]

        try:
            weeks = list(np.arange(35,53+1)) + list(np.arange(1,25+1))
            ilia = ilia.loc[weeks]
        except:
            weeks = list(np.arange(35,52+1)) + list(np.arange(1,25+1))
            ilia = ilia.loc[weeks]

        try:
            weeks = list(np.arange(35,53+1)) + list(np.arange(1,25+1))
            ilib = ilib.loc[weeks]
        except:
            weeks = list(np.arange(35,52+1)) + list(np.arange(1,25+1))
            ilib = ilib.loc[weeks]

        peak_a = np.nanargmax( ilia.to_numpy()[:,:-1], axis=0)
        peak_b = np.nanargmax( ilib.to_numpy()[:,:-1], axis=0)

        past_cases   = cases.iloc[:,:-1].interpolate(axis=0,limit_direction="both")
        target_cases = cases.iloc[:,-1]
        
        cases = np.hstack([past_cases.to_numpy(),target_cases.to_numpy()[:,None]])

        ttl_hosp_constraints = pd.read_csv("./analysis_data/constraint_on_hosps.csv")

        #--map this to the specific state
        from_US_to_state_ttl_hosps = pd.read_csv("./analysis_data/from_US_hosps_to_state_hosps.csv")
        from_US_to_state_ttl_hosps = from_US_to_state_ttl_hosps.loc[from_US_to_state_ttl_hosps.location==location]

        state_hosp_constraints = {}
        state_hosp_constraints["contraint_mean"] = float( from_US_to_state_ttl_hosps["b0"] + from_US_to_state_ttl_hosps["b1"]*float(ttl_hosp_constraints.contraint_mean))
        state_hosp_constraints["constraint_sd"]  = float(np.sqrt( (from_US_to_state_ttl_hosps["b1"]*float(ttl_hosp_constraints.constraint_sd))**2 + from_US_to_state_ttl_hosps["sd"]**2))
        

        #--estimated ILI peak
        ili_peak_estimate = pd.read_csv("./analysis_data/predictions_of_peak_ili.csv")
        ili_peak_estimate = ili_peak_estimate.loc[ili_peak_estimate.location==location]
        
        #--map from ili to hosps
        from_ili_to_hosps = pd.read_csv("./analysis_data/ili_to_peak_hosp_regression.csv")
        from_ili_to_hosps = from_ili_to_hosps.loc[from_ili_to_hosps.location==location]

        #--Y = b0+b1*X+epsilon
        mu_peak  = float(from_ili_to_hosps["b0"] + from_ili_to_hosps["b1"]*ili_peak_estimate["mu"])
        sd_peak  = float(np.sqrt( from_ili_to_hosps["sd"]**2 + (from_ili_to_hosps["b1"]**2)*(ili_peak_estimate["sigma"]**2)))

        #--From Peak A to Peak B
        peak_a_to_b = pd.read_csv("./analysis_data/from_peak_a_to_peak_b.csv")
        peak_a_to_b = peak_a_to_b.loc[peak_a_to_b.location==location]

        peak_hosp_constraints = pd.read_csv("./analysis_data/constraint_on_peak_hosps.csv")

        #--map this to the specific state
        from_US_to_state_peak_hosps = pd.read_csv("./analysis_data/from_US_peak_hosps_to_state_hosps.csv")
        from_US_to_state_peak_hosps = from_US_to_state_peak_hosps.loc[from_US_to_state_peak_hosps.location==location]


        state_hosp_peak_constraints = {}
        state_hosp_peak_constraints["contraint_mean"] = float( from_US_to_state_peak_hosps["b0"] + from_US_to_state_peak_hosps["b1"]*float(peak_hosp_constraints.contraint_mean))
        state_hosp_peak_constraints["constraint_sd"]  = float(np.sqrt( (from_US_to_state_peak_hosps["b1"]*float(peak_hosp_constraints.constraint_sd))**2 + from_US_to_state_peak_hosps["sd"]**2))
        

        if np.isnan(float(peak_a_to_b["b0"])):
           peak_a_to_b = None
            
        model = puca(  y = [cases[:,-1] ]
                     , Y = [cases[:,:-1]]
                     , X = None, anchor = None ).fit(   total_y_target    = float(state_hosp_constraints["contraint_mean"])
                                                      , total_y_sd        = float(state_hosp_constraints["constraint_sd"])
                                                      , peak_time_target  = mu_peak
                                                      , peak_time_sd      = sd_peak
                                                      , peak_temperature  = 0.05
                                                      , peak_y_target     = float(state_hosp_peak_constraints["contraint_mean"])
                                                      , peak_y_sd         = float(state_hosp_peak_constraints["constraint_sd"])
                                                      , peak_a_to_b       = peak_a_to_b)

        forecast = model.forecast(forecast_from_derivative=True)
        yhats    = forecast.squeeze()
      
        #--STORE DATA-----------------------------------------------------
        #---extract quantiles
        quantiles          = np.append(np.append([0.01,0.025],np.arange(0.05,0.95+0.05,0.05)), [0.975,0.99])

        #--WEEKLY INCIDENCE DATA------------------------------------------------------------------------------
        weekly_times            = np.percentile(yhats, quantiles*100, axis=0) #--the -1 is the most recent season
        
        def generate_epiweek_end_dates(start_year, start_week, end_year, end_week):
            end_dates = []
            current_week = Week(start_year, start_week)
            end_week_obj = Week(end_year, end_week)

            while current_week <= end_week_obj:
                # Calculate the Sunday (end of the week)
                end_dates.append(current_week.enddate())
                # Move to the next week
                current_week = current_week + 1

            return end_dates

        # Define the start and end epiweeks for the 20245/2026 season
        start_year          = int(season.split("/")[0])
        start_week          = 35  
        end_year, end_week  = start_year+1, 26 #was 22

        reference_date         = Week.thisweek().enddate() 
        
        # Generate and print all epiweek end dates for the 2025/2026 influenza season
        timepoints = generate_epiweek_end_dates(start_year, start_week, end_year, end_week)
        
        #--add data to dictionary
        forecast_data = {"reference_date"  :[]
                         ,"horizon"        :[]
                         ,"target_end_date":[]
                         ,"output_type_id" :[]
                         ,"value"          :[]}
        
        for forecast_time,d in zip(timepoints, yhats.T):
            fmt = "%Y-%m-%d"
            
            forecast_data["reference_date"].extend( [reference_date.strftime(fmt)]*23 )

            week_from_reference = int((forecast_time - reference_date).days/7)

            weekly_forecast = np.percentile(d, quantiles*100) #--the -1 is the most recent season

            forecast_data["horizon"].extend( [week_from_reference]*23 )

            ted = Week.fromdate(forecast_time).enddate().strftime(fmt)
            forecast_data["target_end_date"].extend([ted]*23)

            forecast_data["output_type_id"].extend( ["{:0.3f}".format(x) for x in quantiles] )
            forecast_data["value"].extend( [ int(x) for x in np.floor(weekly_forecast)] )
            
        weekly_forecast_data = pd.DataFrame(forecast_data)
        weekly_forecast_data["location"]    = location
        weekly_forecast_data["output_type"] = "quantile"
        weekly_forecast_data["target"]      = "wk inc flu hosp"

        columns = ["reference_date","target","horizon","target_end_date","location","output_type","output_type_id","value"]
        weekly_forecast_data = weekly_forecast_data[columns]

        season = season.replace("/","_")

        if location=="US":
            weekly_forecast_data.to_csv("./real_time/shape_plus_deriv_forecasts/forecast_US_{:s}_{:s}.csv".format(season,thisweek))
        else:
            weekly_forecast_data.to_csv("./real_time/shape_plus_deriv_forecasts/forecast_{:s}_{:02d}_{:s}.csv".format(season,int(location),thisweek))

        if location=="US":
            pickle.dump( open("./real_time/shape_plus_deriv_forecasts/forecast_US_{:s}_{:s}.pkl".format(season,thisweek)) )
        else:
            pickle.dump( open("./real_time/shape_plus_deriv_forecasts/forecast_{:s}_{:02d}_{:s}.pkl".format(season,int(location),thisweek)) )

    def tryit(location,season,thisweek,subset,ili_augmented):
        forecast(location,season,thisweek,subset,ili_augmented)
        
    orig_inc = inc_hosps.copy()

    season              = "2026/2027"
    seasons             = [season]
    
    ili_augmented       = ili_augmented.loc[ili_augmented.season.isin(seasons)]
    
    past_inc_hosps      = orig_inc.loc[~orig_inc.season.isin(seasons)]
    inc_hosps           = orig_inc.loc[ orig_inc.season.isin(seasons)] 
    
    #--Cap workers at the physical core count so we never spawn more single-
    #--threaded fits than cores (each fit is pinned to 1 thread via the env
    #--vars at the top of this file). loky backend re-imports this module in
    #--each worker, so those thread caps take effect there too.
    n_jobs = min(20, os.cpu_count() or 1)
    Parallel(n_jobs=n_jobs, backend="loky")( delayed(tryit)(location,season,cutoff,subset,ili_augmented) for (location,season), subset in inc_hosps.groupby(["location","season"]) )
