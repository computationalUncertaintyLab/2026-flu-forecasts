#mcandrew

import sys
import numpy as np
import pandas as pd

if __name__ == "__main__":

    state_to_abbreviation = {
        "Alabama": "al",
        "Alaska": "ak",
        "Arizona": "az",
        "Arkansas": "ar",
        "California": "ca",
        "Colorado": "co",
        "Connecticut": "ct",
        "Delaware": "de",
        "Florida": "fl",
        "Georgia": "ga",
        "Hawaii": "hi",
        "Idaho": "id",
        "Illinois": "il",
        "Indiana": "in",
        "Iowa": "ia",
        "Kansas": "ks",
        "Kentucky": "ky",
        "Louisiana": "la",
        "Maine": "me",
        "Maryland": "md",
        "Massachusetts": "ma",
        "Michigan": "mi",
        "Minnesota": "mn",
        "Mississippi": "ms",
        "Missouri": "mo",
        "Montana": "mt",
        "Nebraska": "ne",
        "Nevada": "nv",
        "New Hampshire": "nh",
        "New Jersey": "nj",
        "New Mexico": "nm",
        "New York": "ny",
        "North Carolina": "nc",
        "North Dakota": "nd",
        "Ohio": "oh",
        "Oklahoma": "ok",
        "Oregon": "or",
        "Pennsylvania": "pa",
        "Puerto Rico" : "pr",
        "Rhode Island": "ri",
        "South Carolina": "sc",
        "South Dakota": "sd",
        "Tennessee": "tn",
        "Texas": "tx",
        "Utah": "ut",
        "Vermont": "vt",
        "Virginia": "va",
        "Washington": "wa",
        "West Virginia": "wv",
        "Wisconsin": "wi",
        "Wyoming": "wy",
        "District of Columbia":"dc",
        "Puerto Rico":"pr",
        "National":"nat"
    }
    ili_data = pd.read_csv("./analysis_data/ili_data_all_states_2021_present.csv")
    #ili_data = pd.read_csv("./analysis_data/ili_data_all_states_2009_present.csv")

    ili_data["state"] = ili_data["state_name"].replace(state_to_abbreviation)

    def addup(x):
        ttl_num_ili      = x.num_ili.sum()
        ttl_num_patients = x.num_patients.sum() 

        ili = 100*(ttl_num_ili /  ttl_num_patients )
        return pd.Series({"ili":float(ili)})
    
    ili_nat = ili_data.groupby(["epiweek","year","week"]).apply(addup).reset_index()
    ili_nat['state'] = "nat"

    ili_data = pd.concat([ili_nat, ili_data])

    def format_location(x):
        if x=="US":
            return x
        else:
            if np.isnan(x):
                return np.nan
            return "{:02d}".format(int(x))
    ili_data["state_fips"] = [format_location(x) for x in ili_data.state_fips.values]
    
    ili_data.to_csv("./analysis_data/ili_data_all_states_2021_present__formatted.csv")
 

