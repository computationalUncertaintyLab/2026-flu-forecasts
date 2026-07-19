#mcandrew

import sys
import numpy as np
import pandas as pd

if __name__ == "__main__":

    #--add epiweek
    lab_data = pd.read_csv("./analysis_data/clinical_and_public_lab_data.csv")
    lab_data["epiweek"] = [ "{:04d}{:02d}".format(x,y)  for (x,y) in zip(lab_data.year.values,lab_data.week.values) ] 

    #--add in state abbreviation lwoer case
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

    lab_data["state"] = lab_data["region"].replace(state_to_abbreviation) 
    lab_data = lab_data.drop(columns = "Unnamed: 0")

    
    lab_data.to_csv("./analysis_data/clinical_and_public_lab_data__formatted.csv", index=False)
