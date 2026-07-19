#--2026-flu-forecasts

![./logo/puca_animal.png](./logo/puca_animal.png)

## Running Real-time forecasts for the 2026-27 season
To run real-time forecasts during the 2026-27 US influenza season (approximately epidemic week 35 of year 2026 to epidemic week 22 of year 2027) you can run the command `make`. 

Make fill run the following routine:
1. Download data
	1. Download ILI data from ILINet
	2. Download Laboratory-confirmed influenza data from ILINet 
	3. Download influenza hospitalizations from NHSN
	4. Download percent of facilities reporting hospitalizations from NHSN

2. Run three component model forecats
	1. Run the first component model forecast (using 20 CPUs in parallel)
	2. Run the second component model forecast (using 20 CPUs in parallel)
	3. Run the third component model forecast (using 20 CPUs in parallel) 

3. Score all of the component models
	1. This will produce a scores.csv file that contains the WIS per component model. 
	2. (RE)compute ensemble weights per location / per horizon 

**Please note** that the puca-semble depends on several seasona-level forecasts that deopend on Southern hemisphere data. These models should be run just once for the season. They can be run with the following command `make one_run_for_season`. 

## Component models
At present (2026-07-19), the puca-semble is a multi-model ensemble of three different component models. Information about each component model can be found in `./model/`.   

Every folder contains the python code used to train and forecast as well as a pdf file that describes the mathemtics/statistics of the model. 

## Ensemble 
XX


## Contact 
Issues, pull requests, etc can be sent to the corresponding author of this repository: tom mcandrew (mcandrew@lehigh.edu). 








