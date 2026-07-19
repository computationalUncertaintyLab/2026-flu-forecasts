#mcandrew

import sys
import numpy as np
import pandas as pd

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    state_to_fips = {
       "AL": 1,  "AK": 2,  "AZ": 4,  "AR": 5,  "CA": 6,
       "CO": 8,  "CT": 9,  "DE": 10, "FL": 12, "GA": 13,
       "HI": 15, "ID": 16, "IL": 17, "IN": 18, "IA": 19,
       "KS": 20, "KY": 21, "LA": 22, "ME": 23, "MD": 24,
       "MA": 25, "MI": 26, "MN": 27, "MS": 28, "MO": 29,
       "MT": 30, "NE": 31, "NV": 32, "NH": 33, "NJ": 34,
       "NM": 35, "NY": 36, "NC": 37, "ND": 38, "OH": 39,
       "OK": 40, "OR": 41, "PA": 42, "RI": 44, "SC": 45,
       "SD": 46, "TN": 47, "TX": 48, "UT": 49, "VT": 50,
       "VA": 51, "WA": 53, "WV": 54, "WI": 55, "WY": 56,
        "DC": 11, "AS": 60, "GU": 66, "MP": 69, "PR": 72, "VI": 78}
    state_to_fips = { x:"{:02d}".format(y) for x,y in state_to_fips.items() }
    state_to_fips["USA"] = "US"
    
    nhsn = pd.read_csv("https://data.cdc.gov/resource/mpgq-jmmr.csv?$limit=999999")
    #nhsn = pd.read_csv("https://data.cdc.gov/resource/ua7e-t2fy.json?$limit=999999")

    nhsn["location"] = nhsn.jurisdiction.replace(state_to_fips)

    pct_hosps_reporting = nhsn[ ["weekendingdate","location",'totalconffluhosppatsperc'] ]
    pct_hosps_reporting  = pct_hosps_reporting.rename(columns = {"totalconffluhosppatsperc":"pct_hosp"}) 

    #--from week to epiweek
    pct_hosps_reporting["date"] = [ datetime.strptime(x.split(".000")[0],"%Y-%m-%dT%H:%M:%S").strftime("%Y-%m-%d")  for x in pct_hosps_reporting.weekendingdate.values]


    def fromdate2MMWR(row):
        from datetime import datetime, timedelta
        from epiweeks import Week
        dt = row.date
        w = Week.fromdate( datetime.strptime(dt,"%Y-%m-%d") )
        row["MMWRYR"] = w.year
        row["MMWRWK"] = w.week
        return row

    pct_hosps_reporting = pct_hosps_reporting.apply(fromdate2MMWR,1)
    
    pct_hosps_reporting.to_csv("./analysis_data/pct_hospital_reporting.csv",index=False)

   

