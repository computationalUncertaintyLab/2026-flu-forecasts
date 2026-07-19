#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob
from datetime import datetime, timedelta

if __name__ == "__main__":

    d = []
    for fil in glob("./real_time/shape_plus_deriv_forecasts/*.csv"):
        ref_date = fil.split("_")[-1].split(".")[0]
        subset = pd.read_csv(fil)
        subset["reference_date"] = ref_date

        def update_horizon(row):
            ref_date = datetime.strptime(row.reference_date , "%Y-%m-%d")
            target   = datetime.strptime(row.target_end_date, "%Y-%m-%d")

            return int((target - ref_date).days/7)
            
        subset["horizon"] = subset.apply(update_horizon,1)
        subset["model"] = "level"
        
        d.append( subset )
        
    experimental_forecasts_1 = pd.concat(d)
    
    d = []
    for fil in glob("./deriv_shape_forecasts/*.csv"):
        ref_date = fil.split("_")[-1].split(".")[0]
        subset = pd.read_csv(fil)
        subset["reference_date"] = ref_date

        def update_horizon(row):
            ref_date = datetime.strptime(row.reference_date , "%Y-%m-%d")
            target   = datetime.strptime(row.target_end_date, "%Y-%m-%d")

            return int((target - ref_date).days/7)
            
        subset["horizon"] = subset.apply(update_horizon,1)
        subset["model"] = "deriv"
        
        d.append( subset )
        
    experimental_forecasts_2 = pd.concat(d)

    d = []
    for fil in glob("./KF_forecasts/*.csv"):
        ref_date = fil.split("_")[-1].split(".")[0]
        subset = pd.read_csv(fil)
        subset["reference_date"] = ref_date

        def update_horizon(row):
            ref_date = datetime.strptime(row.reference_date , "%Y-%m-%d")
            target   = datetime.strptime(row.target_end_date, "%Y-%m-%d")

            return int((target - ref_date).days/7)
            
        subset["horizon"] = subset.apply(update_horizon,1)
        subset["model"] = "KF"
        
        d.append( subset )
        
    KF_forecasts = pd.concat(d)
    
    #MM_forecasts  = pd.read_csv("./multi_model_ensemble_weights/adaptive_wis_ensemble_opt.csv")
    #MM_forecasts["model"] = "MM"

    all_forecasts = pd.concat([experimental_forecasts_1,experimental_forecasts_2, KF_forecasts])

    inc_hosps  = pd.read_csv("./data/target-data/target-hospital-admissions.csv")
    inc_hosps  = inc_hosps.loc[ (inc_hosps["date"]>="2021-10-09")  ]

    def format(x):
        if x=="US":
            return x
        return "{:02d}".format(int(x))

    all_forecasts["location"] = [format(x) for x in all_forecasts.location.values]

    inc_hosps = inc_hosps.rename(columns = {"value":"observed"})
    all_forecasts = all_forecasts.merge( inc_hosps, left_on = ["target_end_date","location"], right_on = ["date","location"] )

    all_forecasts = all_forecasts.rename(columns = {"value":"predicted"})
    
    columns =  ["reference_date","horizon","target_end_date","location","output_type_id",  "predicted","model", "observed"]
    all_forecasts = all_forecasts[columns]
    all_forecasts = all_forecasts.loc[all_forecasts.horizon>0]
    
    all_forecasts.to_csv("./multi_model_ensemble_weights/component_model_forecasts.csv",index=False)
